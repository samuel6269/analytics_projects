import random as rand
import numpy as np  		  	   		     		  		  		    	 		 		   		 		  

class QLearner(object):  		  	   		     		  		  		    	 		 		   		 		  
    """  		  	   		     		  		  		    	 		 		   		 		  
    This is a Q learner object.  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
    :param num_states: The number of states to consider.  		  	   		     		  		  		    	 		 		   		 		  
    :type num_states: int  		  	   		     		  		  		    	 		 		   		 		  
    :param num_actions: The number of actions available..  		  	   		     		  		  		    	 		 		   		 		  
    :type num_actions: int  		  	   		     		  		  		    	 		 		   		 		  
    :param alpha: The learning rate used in the update rule. Should range between 0.0 and 1.0 with 0.2 as a typical value.  		  	   		     		  		  		    	 		 		   		 		  
    :type alpha: float  		  	   		     		  		  		    	 		 		   		 		  
    :param gamma: The discount rate used in the update rule. Should range between 0.0 and 1.0 with 0.9 as a typical value.  		  	   		     		  		  		    	 		 		   		 		  
    :type gamma: float  		  	   		     		  		  		    	 		 		   		 		  
    :param rar: Random action rate: the probability of selecting a random action at each step. Should range between 0.0 (no random actions) to 1.0 (always random action) with 0.5 as a typical value.  		  	   		     		  		  		    	 		 		   		 		  
    :type rar: float  		  	   		     		  		  		    	 		 		   		 		  
    :param radr: Random action decay rate, after each update, rar = rar * radr. Ranges between 0.0 (immediate decay to 0) and 1.0 (no decay). Typically 0.99.  		  	   		     		  		  		    	 		 		   		 		  
    :type radr: float  		  	   		     		  		  		    	 		 		   		 		  
    :param dyna: The number of dyna updates for each regular update. When Dyna is used, 200 is a typical value.  		  	   		     		  		  		    	 		 		   		 		  
    :type dyna: int  		  	   		     		  		  		    	 		 		   		 		  
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		     		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		     		  		  		    	 		 		   		 		  
    """  		  	   		     		  		  		    	 		 		   		 		  
    def __init__(  		  	   		     		  		  		    	 		 		   		 		  
        self,  		  	   		     		  		  		    	 		 		   		 		  
        num_states=100,  		  	   		     		  		  		    	 		 		   		 		  
        num_actions=4,  		  	   		     		  		  		    	 		 		   		 		  
        alpha=0.2,  		  	   		     		  		  		    	 		 		   		 		  
        gamma=0.9,  		  	   		     		  		  		    	 		 		   		 		  
        rar=0.5,  		  	   		     		  		  		    	 		 		   		 		  
        radr=0.99,  		  	   		     		  		  		    	 		 		   		 		  
        dyna=0,  		  	   		     		  		  		    	 		 		   		 		  
        verbose=False,  		  	   		     		  		  		    	 		 		   		 		  
    ):  		  	   		     		  		  		    	 		 		   		 		  
        """  		  	   		     		  		  		    	 		 		   		 		  
        Constructor method  		  	   		     		  		  		    	 		 		   		 		  
        """  		  	   		     		  		  		    	 		 		   		 		  
        self.verbose = verbose
        self.num_actions = num_actions  		  	   		     		  		  		    	 		 		   		 		  
        self.s = 0  		  	   		     		  		  		    	 		 		   		 		  
        self.a = 0
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.num_states = num_states
        self.Q = np.zeros((num_states,num_actions))
        #self.T = np.full((num_states,num_actions,num_states),fill_value = 0.00001)
        self.T =np.zeros((num_states,num_actions), dtype=int)
        self.R = np.zeros((num_states,num_actions))
        #self.R = np.full((num_states,num_actions),fill_value = -0.001)
        self.samples = []
  		  	   		     		  		  		    	 		 		   		 		  
    def querysetstate(self, s):  		  	   		     		  		  		    	 		 		   		 		  
        """  		  	   		     		  		  		    	 		 		   		 		  
        Update the state without updating the Q-table  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
        :param s: The new state  		  	   		     		  		  		    	 		 		   		 		  
        :type s: int  		  	   		     		  		  		    	 		 		   		 		  
        :return: The selected action  		  	   		     		  		  		    	 		 		   		 		  
        :rtype: int  		  	   		     		  		  		    	 		 		   		 		  
        """  		  	   		     		  		  		    	 		 		   		 		  
        self.s = s
        rand_action_p = rand.random()
        if rand_action_p <= self.rar:
            action = rand.randint(0, self.num_actions - 1)
        else:
            action = np.argmax(self.Q[self.s])

        self.rar = self.rar * self.radr
        self.a = action
        if self.verbose:
            print(f"s = {s}, a = {action}")  		  	   		     		  		  		    	 		 		   		 		  
        return action  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
    def query(self, s_prime, r):  		  	   		     		  		  		    	 		 		   		 		  
        """  		  	   		     		  		  		    	 		 		   		 		  
        Update the Q table and return an action  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
        :param s_prime: The new state  		  	   		     		  		  		    	 		 		   		 		  
        :type s_prime: int  		  	   		     		  		  		    	 		 		   		 		  
        :param r: The immediate reward  		  	   		     		  		  		    	 		 		   		 		  
        :type r: float  		  	   		     		  		  		    	 		 		   		 		  
        :return: The selected action  		  	   		     		  		  		    	 		 		   		 		  
        :rtype: int  		  	   		     		  		  		    	 		 		   		 		  
        """
        rand_action_p = rand.random()
        if rand_action_p <= self.rar:
            action = rand.randint(0, self.num_actions - 1)
            best_action = np.argmax(self.Q[s_prime])
            self.Q[self.s, self.a] = (1 - self.alpha) * self.Q[self.s, self.a] + self.alpha * (r + self.gamma * self.Q[s_prime, best_action])
        else:
            action = np.argmax(self.Q[s_prime])
            self.Q[self.s,self.a] = (1-self.alpha)*self.Q[self.s,self.a] + self.alpha * (r + self.gamma*self.Q[s_prime,action])

        #Learn T and R models
        #self.T[self.s,self.a,s_prime] += 1
        #self.T[self.s, self.a, s_prime] /= np.sum(self.T[self.s,self.a])
        #self.R[self.s,self.a] = (1-self.alpha) * self.R[self.s,self.a] + self.alpha * r

        if (self.s,self.a) not in self.samples:
            self.samples.append((self.s,self.a))
        self.T[self.s, self.a] = s_prime
        self.R[self.s, self.a] = r

        #start dyna procedure

        # for i in range(self.dyna):
        #     (s,a) = rand.sample(self.samples, 1)[0]
        #     #s_dyna_prime = np.argmax(self.T[s,a])
        #     s_dyna_prime = self.T[s, a]
        #     r = self.R[s,a]
        #     self.Q[s, a] = (1 - self.alpha) * self.Q[s, a] + self.alpha * (
        #                      r + self.gamma * self.Q[s_dyna_prime, np.argmax(self.Q[s_dyna_prime])])

        #lets vectorize this
        if self.dyna > 0:
            sa_list = rand.choices(self.samples, k=self.dyna)
            s = [i[0] for i in sa_list]
            a = [i[1] for i in sa_list]
            s_dyna_prime = self.T[s,a]
            r = self.R[s,a]
            self.Q[s, a] = (1 - self.alpha) * self.Q[s, a] + self.alpha * (
                    r + self.gamma * self.Q[s_dyna_prime, np.argmax(self.Q[s_dyna_prime], axis =1)])

        if self.verbose:  		  	   		     		  		  		    	 		 		   		 		  
            print(f"s = {s_prime}, a = {action}, r={r}")
        self.s = s_prime
        self.a = action
        self.rar = self.rar * self.radr
        return action