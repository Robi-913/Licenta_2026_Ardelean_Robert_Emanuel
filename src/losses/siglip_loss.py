import torch
import torch.nn as nn


class SigLIPLoss(nn.Module):
    # nn.Module e radacina pt tot ce inseamna straturi, modele si functii de loss in pytorch

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        # Binary Cross Entropy este functia de loss care ne spune cat de departe este o predictie de valoarea corecta(0/1)
        # nu facem media pt ca avem nevoie de lossul individual pt fiecare pereche imagine-text din batch

    def forward(self, img_emb, txt_emb, logit_scale):

        """
        :param img_emb: embeddingul imaginei (vector de dimensiune fixa care reprezinta imaginea)
        :param txt_emb: embeddingul textului (vector de dimensiune fixa care reprezinta textul)
        :param logit_scale: temperatura pt scalare (cat de mult vrem sa accentuam diferentele dintre perechile corecte si cele incorecte)

        embeddingul = [B, D] -> B = batch size, D = dimensiunea vectorului

        :return: loss.mean
        """

        batch_size = img_emb.shape[0] # sizeul batchului

        similarity = img_emb @ txt_emb.T # @ -> dot product pt a crea matricea de similaritate

        # ex:
        #     img_emb (2 imagini, 3 numere fiecare):
        #     imagine_0 = [0.5, 0.3, 0.1]
        #     imagine_1 = [0.1, 0.8, 0.2]
        #
        #     txt_emb (2 texte, 3 numere fiecare):
        #     text_0 = [0.4, 0.2, 0.1]
        #     text_1 = [0.2, 0.7, 0.3]
        #
        #               text_0  text_1
        #     imagine_0 [0.27   0.34]
        #     imagine_1 [0.22   0.64]


        scaled = similarity * logit_scale.exp()
        # fortam modelul sa amplifice diferentele valorile
        # ex:
        #     scale = 10
        #               text_0  text_1
        #     imagine_0 [2.70   3.40]
        #     imagine_1 [2.20   6.40]

        target = torch.eye(batch_size, device=img_emb.device) # cream matricea identitate

        loss = self.bce(scaled, target)
        # scaled:                     target:
        # [[2.7,  3.4],               [[1, 0],
        # [2.2,  6.4]]                [0, 1]]
        #
        # pozitia [0,0]: scor 2.7, tinta 1 -> e corect, deci loss mic
        # pozitia [0,1]: scor 3.4, tinta 0 -> e prea mare pt o potrivire gresita, deci loss mare(2.7 e mai mic decat 3.4)
        # pozitia [1,0]: scor 2.2, tinta 0 -> e maricel pt ceva gresit, deci loss mediu
        # pozitia [1,1]: scor 6.4, tinta 1 -> potrivire excelenta, deci loss foarte mic

        return loss.mean() #calcula media lossurilor




def contrastive_accuracy(img_emb, txt_emb):
    # masoara cat de bine asociaza modelul imaginile cu textele

    batch_size = img_emb.shape[0]
    sim = img_emb @ txt_emb.T

    correct = torch.arange(batch_size, device=sim.device)
    # calsificam ce imagine se leaga de ce text
    # correct = [0, 1] deoarece in exemplu nostru:
    # imaginea 0 trebuie sa aleaga textul corespunzator adica 2.7
    # imaginea 1 trebuie sa aleaga textul corespunzator adica 6.4

    i2t = (sim.argmax(dim=1) == correct).sum().item() / batch_size * 100.0
    t2i = (sim.argmax(dim=0) == correct).sum().item() / batch_size * 100.0
    # modelul ia vectorul corect prezis de el si verificam corectitatea lui
    # i2t: pt fiecare imagine, verificam daca textul cu scorul cel mai mare este cel corect
    # t2i: pt fiecare text, verificam daca imaginea cu scorul cel mai mare este cea corecta
    #
    #           text_0  text_1
    # imagine_0 [0.27   0.34]
    # imagine_1 [0.22   0.64]
    #
    # coloana 0: max e 0.27, pe poziția 0
    # coloana 1: max e 0.64, pe poziția 1
    #
    # argmax: [0, 1]
    # correct: [0, 1]
    # compară: [True, True]
    # sum: 2
    # procent: 2/2 * 100 = 100%
    # correct = [0, 1] -> mereu fix, răspunsul corect
    # argmax = [1, 1] -> ce a ales modelul (depinde de valori)
    # compară: [False, True] -> unde a nimerit

    return i2t, t2i