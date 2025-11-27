from player_controller_hmm import PlayerControllerHMMAbstract
from constants import *
import functions as fn 
import random


class PlayerControllerHMM(PlayerControllerHMMAbstract):
    def init_parameters(self):

        # Settings
        self.N = 3                                          # Hidden States   
        self.M = N_EMISSIONS                                # This the number of Fish Moments

        # initial uniform parameters
        self.A_o = [[1.0/self.N]*self.N for _ in range(self.N)]
        self.B_o = [[1.0/self.M]*self.M for _ in range(self.N)]
        self.pi_o = [1.0/self.N]*self.N

        # Loop through each species and iteratete through the models
        self.models = []
        for _ in range(N_SPECIES):
            A_copy = []
            for row in self.A_o:
                new_row = row[:]        
                A_copy.append(new_row)

            # Copy B0 into a new matrix B
            B_copy = []
            for row in self.B_o:
                new_row = row[:]       
                B_copy.append(new_row)

            # Copy pi0
            pi_copy = self.pi_o[:]      

            # Append model dictionary
            model = {
                "A": A_copy,
                "B": B_copy,
                "pi": pi_copy,
                "trained": False
            }

            self.models.append(model)


        # Observation lists for each fish
        self.fish_obs = [[] for _ in range(N_FISH)]

        self.done = [False]*N_FISH

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

        best_race = None
        best_fish = None

        # Append the new observations that were made for each fish!!!!
        for fish in range(N_FISH):
            if self.done[fish]:
                continue
            if not self.done[fish]:
                self.fish_obs[fish].append(observations[fish])
    
            # Now we need to check each trained species

            max_prob = float('-inf')
            best_race = None        # I swear I'm not racist
            best_fish = None

            for species_id in range(N_SPECIES):
                model = self.models[species_id]
                
                #Run the fcking forward algorithim with model with all fish
                final_prob = fn.forward_algorithm(self.models["A"], self.models["B"], self.models["pi"], observations)

                if(final_prob > max_prob):
                    best_race = species_id
                    best_fish = fish
                
        return best_fish, best_race

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

        # This fish will not produce more data, so that mark in the sequence will be complete
        self.done[fish_id] = True

        # Storing the fish sequence over time as the game progresse
        seq = self.fish_obs[fish_id]

        # This will validate if the deque
        if len(seq) < 5:
            return

        # Train the correct species HMM with Baum-Welch
        A, B, pi, log_liklihood = fn.baum_welch(seq, self.N, self.M, self.A_o, self.B_o, self.pi_o, tol=1e-4, max_iter=100)

        # Save the Train models into a dictionary!!!!
        model = self.models[true_type]
        model["A"] = A
        model["B"] = B
        model["pi"] = pi
        model["trained"] = True

        return

'''
        self.models = [
            {"A":..., "B":..., "pi":..., "trained":False},   # species 0
            {"A":..., "B":..., "pi":..., "trained":False},   # species 1
            ...
        ]
'''