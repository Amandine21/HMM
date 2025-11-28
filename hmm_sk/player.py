from player_controller_hmm import PlayerControllerHMMAbstract
from constants import *
import functions as fn 
import sys
import time


class PlayerControllerHMM(PlayerControllerHMMAbstract):
    def init_parameters(self):
        # --- HMM-Grundparameter ---
        self.N = 3                  # number of hidden states
        self.M = N_EMISSIONS        # number of emissions

        # uniforme Startparameter
        self.A0 = [[1.0 / self.N] * self.N for _ in range(self.N)]
        self.B0 = [[1.0 / self.M] * self.M for _ in range(self.N)]
        self.pi0 = [1.0 / self.N] * self.N

        # ein HMM pro Species
        self.models = []
        for _ in range(N_SPECIES):
            A_copy = [row[:] for row in self.A0]
            B_copy = [row[:] for row in self.B0]
            pi_copy = self.pi0[:]
            self.models.append({
                "A": A_copy,
                "B": B_copy,
                "pi": pi_copy,
            })

        # Beobachtungen pro Fisch
        self.fish_obs = [[] for _ in range(N_FISH)]
        # Fische, die fertig sind (bereits revealed)
        self.done = [False] * N_FISH
        # Zähle, wie oft wir pro Species trainiert haben
        self.train_count = [0] * N_SPECIES

        self.guess_count = 0

        # Heuristik-Parameter
        self.MAX_TRAIN_PER_SPECIES = 10  # max. Trainingsläufe pro Species
        self.MAX_TRAIN_LEN = 180         # nur die letzten K Beobachtungen zum Trainieren nutzen
        self.BW_MAX_ITERS = 50           # Iterationen für Baum-Welch



    # -----------------------------------------------------------

    def guess(self, step, observations):
        """
        This method gets called on every iteration, providing observations.
        Here the player should process and store this information,
        and optionally make a guess by returning a tuple containing the fish index and the guess.
        :param step: iteration number
        :param observations: a list of N_FISH observations, encoded as integers
        :return: None or a tuple (fish_id, fish_type)
        """

        
        # nur Daten sammeln
        for f in range(N_FISH):
            if not self.done[f]:
                self.fish_obs[f].append(observations[f])

        # --- Progressive guessing schedule ---

        if step < 110:
            return None

        best_prob = float('-inf')
        best_fish = None
        best_species = None

        # Für jeden noch „aktiven“ Fisch Likelihood ausrechnen
        for f in range(N_FISH):
            if self.done[f]:
                continue

            seq = self.fish_obs[f]

            # Likelihood unter allen Species-HMMs berechnen
            for species_id in range(N_SPECIES):
                model = self.models[species_id]
                A, B, pi = model["A"], model["B"], model["pi"]

                prob = fn.forward_algorithm(A, B, pi, seq)  # log-Likelihood

                if prob > best_prob:
                    best_prob = prob
                    best_fish = f
                    best_species = species_id

        # Noch keine sinnvolle Entscheidung möglich
        if best_fish is None:
            return None
   
        self.guess_count += 1

        #print(f"Guesses made: {self.guess_count}")

        return best_fish, best_species

    # -----------------------------------------------------------

    def reveal(self, correct, fish_id, true_type):
        """
        This methods gets called whenever a guess was made.
        It informs the player about the guess result
        and reveals the correct type of that fish.
        :param correct: tells if the guess was correct
        :param fish_id: fish's index
        :param true_type: the correct type of the fish
        :return:
        """

        # Dieser Fisch liefert ab jetzt keine neuen Daten mehr
        self.done[fish_id] = True

        seq = self.fish_obs[fish_id]


        # 2) Pro Species nur ein paar Mal trainieren,
        #    um Zeit zu sparen
        if self.train_count[true_type] >= self.MAX_TRAIN_PER_SPECIES:
            return

        # 3) Optional nur die letzten K Beobachtungen verwenden
        if len(seq) > self.MAX_TRAIN_LEN:
            train_seq = seq[-self.MAX_TRAIN_LEN:]
        else:
            train_seq = seq

        model = self.models[true_type]
        A = [row[:] for row in model["A"]]
        B = [row[:] for row in model["B"]]
        pi = model["pi"][:]

        # Baum-Welch (mit wenigen Iterationen)
        A_new, B_new, pi_new, _ = fn.baum_welch(
            A, B, pi, train_seq, self.BW_MAX_ITERS
        )

        # Parameter aktualisieren
        model["A"] = A_new
        model["B"] = B_new
        model["pi"] = pi_new

        self.train_count[true_type] += 1

        return


'''
        self.models = [
            {"A":..., "B":..., "pi":..., "trained":False},   # species 0
            {"A":..., "B":..., "pi":..., "trained":False},   # species 1
            ...
        ]
'''