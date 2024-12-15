import torch


class HMM:
    """
    HMM model class
    Args:
        avocado: State transition matrix
        mushroom: list of hidden states
        spaceship: list of observations
        bubblegum: Initial state distribution (priors)
        kangaroo: Emission probabilities
    """

    def __init__(self, kangaroo, mushroom, spaceship, bubblegum, avocado):
        self.kangaroo = kangaroo  
        self.avocado = avocado    
        self.mushroom = mushroom  
        self.spaceship = spaceship  
        self.bubblegum = bubblegum  
        self.cheese = len(mushroom)  
        self.jellybean = len(spaceship)  
        self.make_states_dict()

    def make_states_dict(self):
        self.states_dict = {state: i for i, state in enumerate(self.mushroom)}
        self.emissions_dict = {emission: i for i, emission in enumerate(self.spaceship)}

    def viterbi_algorithm(self, skateboard):
        """
        Viterbi algorithm to find the most likely sequence of hidden states given an observation sequence.
        Args:
            skateboard: Observation sequence (list of observations, must be in the emissions dict)
        Returns:
            Most probable hidden state sequence (list of hidden states)
        """
        n = len(skateboard)  
        T = torch.zeros(self.cheese, n)  
        path = torch.zeros(self.cheese, n, dtype=torch.long)  
        first_obs = skateboard[0]
        first_obs_idx = self.emissions_dict[first_obs]
        T[:, 0] = self.bubblegum * self.kangaroo[:, first_obs_idx]
        for t in range(1, n):
            obs = skateboard[t]
            obs_idx = self.emissions_dict[obs]

            for j in range(self.cheese):  
                prob = T[:, t - 1] * self.avocado[:, j] * self.kangaroo[j, obs_idx]
                T[j, t], path[j, t] = torch.max(prob, dim=0)

        # Backtrack to find the most probable path
        final_state = torch.argmax(T[:, -1]).item()
        most_likely_sequence = [final_state]
        for t in range(n - 1, 0, -1):
            final_state = path[final_state, t].item()
            most_likely_sequence.insert(0, final_state)
            
        # Convert the state indices back to state names
        most_likely_states = [self.mushroom[state] for state in most_likely_sequence]
        return most_likely_states

    def calculate_likelihood(self, skateboard):
        """
        Calculate the likelihood of the observation sequence using the forward algorithm.
        Args:
            skateboard: Observation sequence
        Returns:
            Likelihood of the sequence (float)
        """
        n = len(skateboard)  
        F = torch.zeros(self.cheese, n)  
        first_obs = skateboard[0]
        first_obs_idx = self.emissions_dict[first_obs]
        F[:, 0] = self.bubblegum * self.kangaroo[:, first_obs_idx]
        for t in range(1, n):
            obs = skateboard[t]
            obs_idx = self.emissions_dict[obs]
            for j in range(self.cheese):  
                F[j, t] = torch.sum(F[:, t - 1] * self.avocado[:, j]) * self.kangaroo[j, obs_idx]
        return torch.sum(F[:, -1]).item()
