from player_controller_hmm import PlayerControllerHMMAbstract
from constants import *
import functions as fn 
import sys
import time


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


        best_prob = float('-inf')
        best_race = None        # I swear I'm not racist
        best_fish = None


        if(step < 40):
            return

        # Append the new observations that were made for each fish!!!!
        for fish in range(N_FISH):
            if self.done[fish]:
                continue
            if not self.done[fish]:
                self.fish_obs[fish].append(observations[fish])
    
            # Now we need to check each trained species

            for species_id in range(N_SPECIES):
                model = self.models[species_id]
                
                A = model["A"]
                B = model["B"]
                pi = model["pi"]

                prob = fn.forward_algorithm(A, B, pi, self.fish_obs[fish])
                #Run the fcking forward algorithim with model with all fish

                if(prob > best_prob):
                    best_prob = prob
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

        # This fish will not produce more data
        self.done[fish_id] = True

        # Full observation sequence of this fish
        seq = self.fish_obs[fish_id]

        # If sequence is too short, don't bother training
        if len(seq) < 5:
            return

        # Get current model for this species
        model = self.models[true_type]

        A = [row[:] for row in model["A"]]
        B = [row[:] for row in model["B"]]
        pi = model["pi"][:]

        # Train/refine HMM for this species with Baum–Welch
                    
        A_new, B_new, pi_new, log_likelihood = fn.baum_welch(A, B, pi, seq, 5)

        # Save updated parameters
        model["A"] = A_new
        model["B"] = B_new
        model["pi"] = pi_new

        return


'''
        self.models = [
            {"A":..., "B":..., "pi":..., "trained":False},   # species 0
            {"A":..., "B":..., "pi":..., "trained":False},   # species 1
            ...
        ]
'''