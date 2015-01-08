from pylab import *
import numpy
from time import sleep
from Math import pi
import matplotlib.pyplot as plt
plt.ion()


class Gridworld:

    def __init__(self, epsilon=0.5, lamb=0.95, sigma=0.05, eta=0.005, gamma=0.95):

        # Grid size
        self.N = 20

        # Exploration-exploitation tradeoff
        self.eta = eta

        # Eligibility trace decay
        self.lamb = lamb

        # Activity standard deviation
        self.sigma = sigma

        # Learning rate
        self.eta = eta

        # Reward discount
        self.gamma = gamma

        # Reward location
        self.reward_position = (0.8, 0.8)

        # Reward administered t the target location and when bumping into walls
        self.reward_at_target = 10.
        self.reward_at_wall   = -2.

        # Starting position of our rat
        self.x_position = 0.1
        self.y_position = 0.1

        # Possible direction
        self.direction = lambda a: 2 * pi * a / 8.

        # Initialize the Q-values etc.
        self._init_run()

    ############################################################################
    # Private methods
    ############################################################################

    def _init_run():
        """
        Initialize the Q-values, eligibility trace, position etc.
        """
        # initialize the Q-values and the eligibility trace
        self.Q = 0.01 * numpy.random.rand(self.N, self.N, 4) + 0.1
        self.e = numpy.zeros((self.N, self.N, 4))

        # list that contains the times it took the agent to reach the target for all trials
        # serves to track the progress of learning
        self.latency_list = []

        # initialize the state and action variables
        self.x_position = None
        self.y_position = None
        self.action = None


    def _run_trial(self,visualize=False):
        """
        Run a single trial on the gridworld until the agent reaches the reward position.
        Return the time it takes to get there.

        Options:
        visual: If 'visualize' is 'True', show the time course of the trial graphically
        """

        # initialize the latency (time to reach the target) for this trial
        latency = 0.

        # start the visualization, if asked for
        if visualize:
            self._init_visualization()

        # run the trial
        self._choose_action()
        while not self._arrived():
            self._update_state()
            self._choose_action()
            self._update_Q()
            if visualize:
                self._visualize_current_state()

            latency = latency + 1

        if visualize:
            self._close_visualization()
        return latency


    def _update_Q(self):
        """
        Update the current estimate of the Q-values according to SARSA.
        """
        # update the eligibility trace
        self.e = self.lambda_eligibility * self.e
        self.e[self.x_position_old, self.y_position_old,self.action_old] += 1.

        # update the Q-values
        if self.action_old != None:
            self.Q +=     \
                self.eta * self.e *\
                (self._reward()  \
                - ( self.Q[self.x_position_old,self.y_position_old,self.action_old] \
                - self.gamma * self.Q[self.x_position, self.y_position, self.action] )  )

    def _choose_action(self):
        """
        Choose the next action based on the current estimate of the Q-values.
        The parameter epsilon determines, how often agent chooses the action
        with the highest Q-value (probability 1-epsilon). In the rest of the cases
        a random action is chosen.
        """
        self.action_old = self.action
        if numpy.random.rand() < self.epsilon:
            self.action = numpy.random.randint(8)
        else:
            self.action = argmax(self.Q[self.x_position,self.y_position,:])

    def _arrived(self):
        """
        Check if the agent has arrived.
        """
        return  (self.x_position - self.reward_position[0]) ^ 2 + \
                (self.y_position - self.reward_position[1]) ^ 2 < 0.01

    def _reward(self):
        """
        Evaluates how much reward should be administered when performing the
        chosen action at the current location
        """
        if self._arrived():
            return self.reward_at_target

        if self._wall_touch:
            return self.reward_at_wall
        else:
            return 0.

    def _update_state(self):
        """
        Update the state according to the old state and the current action.
        """
        # remember the old position of the agent
        self.x_position_old = self.x_position
        self.y_position_old = self.y_position

        # update the agents position according to the action
        # move to the down?
        if self.action == 0:
            self.x_position += 1
        # move to the up
        elif self.action == 1:
            self.x_position -= 1
        # move right?
        elif self.action == 2:
            self.y_position += 1
        # move left?
        elif self.action == 3:
            self.y_position -= 1
        else:
            print "There must be a bug. This is not a valid action!"

        # check if the agent has bumped into a wall.
        if self._is_wall():
            self.x_position = self.x_position_old
            self.y_position = self.y_position_old
            self._wall_touch = True
        else:
            self._wall_touch = False
