from pylab import *
import numpy
from time import sleep
from math import pi, exp, cos, sin, pow, sqrt
import matplotlib.pyplot as plt
plt.ion()


class Gridworld:

    def __init__(self, epsilon=0.5, lambda_eligibility=0.95, sigma=0.05, eta=0.005, gamma=0.95):

        # Grid size
        self.N = 20

        # Exploration-exploitation tradeoff
        self.epsilon = epsilon
        # e = 0 => full exploitation
        # e = 1 => full exploration

        # Eligibility trace decay
        self.lambda_eligibility = lambda_eligibility

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

        # Length of movement
        self.l = 0.03

        # Initialize the weights etc.
        self._init_run()


    def run(self, N_trials=10, N_runs=1):
        self.N_trials = N_trials
        self.N_runs = N_runs
        self.epsilon_start = self.epsilon

        self.latencies = zeros(N_trials)
        self.rewards = zeros(N_trials)

        for run in range(N_runs):
            print "##############"
            print "##### Run:", run
            print "##############"
            self._init_run()
            latencies, rewards = self._learn_run(N_trials=N_trials)
            self.latencies += latencies / N_runs
            self.rewards += rewards / N_runs


    def reset_pos(self):
        self.x_position = 0.1
        self.y_position = 0.1

    def epsilon_for(self, x):
        return max(0, 0.8 - 0.03 * x+0.0003 * (x**2) - 1.02*10**(-6) *x ** 3)


    def reset(self):
        """
        Reset the weigths (and the latency_list).

        Instant amnesia -  the agent forgets everything he has learned before
        """
        self.w = numpy.random.rand(self.N * self.N, 8)
        self.e = numpy.zeros((self.N * self.N, 8))
        self.latency_list = []
        self.reward_list = []
        self.reset_pos()
        self.action = None


    def learning_curve(self, log=False, filter=1.):
        """
        Show a running average of the time it takes the agent to reach the target location.

        Options:
        filter=1. : timescale of the running average.
        log    : Logarithmic y axis.
        """
        fig = figure()
        xlabel('trials')
        ylabel('time to reach target')
        latencies = array(self.latency_list)
        # calculate a running average over the latencies with a averaging time 'filter'
        for i in range(1,latencies.shape[0]):
            latencies[i] = latencies[i-1] + (latencies[i] - latencies[i-1])/float(filter)

        if not log:
            plot(latencies)
        else:
            semilogy(latencies)

        fig.savefig('learning_curve_runs_%s_trials_%s_epsilon_%s.png' % (self.N_runs, self.N_trials, self.epsilon_start))
    def navigation_map(self):
        """
            Plot the direction with the highest Q-value for every position.
            Useful only for small gridworlds, otherwise the plot becomes messy.
            """
        self.x_direction = numpy.zeros((self.N,self.N))
        self.y_direction = numpy.zeros((self.N,self.N))

        self.actions = numpy.zeros((self.N,self.N))

        for i in range(20):
            for j in range(20):
                sx = i * (1 / 19.)
                sy = j * (1 / 19.)
                actions = []
                for a in range(8):
                    actions.append(self._Q(sx, sy, a))
                self.actions[i,j] = argmax(actions)



        #self.actions = argmax(self.Q[:,:,:],axis=2)
        self.y_direction[self.actions==0] = 0
        self.y_direction[self.actions==1] = 1 #.5
        self.y_direction[self.actions==2] = 1.
        self.y_direction[self.actions==3] = 1 #.5
        self.y_direction[self.actions==4] = 0
        self.y_direction[self.actions==5] = -1 #-.5
        self.y_direction[self.actions==6] = -1
        self.y_direction[self.actions==7] = -1 #-.5

        self.x_direction[self.actions==0] = 1.
        self.x_direction[self.actions==1] = 1 #0.5
        self.x_direction[self.actions==2] = 0.
        self.x_direction[self.actions==3] = -1 #-.5
        self.x_direction[self.actions==4] = -1.
        self.x_direction[self.actions==5] = -1 #-.5
        self.x_direction[self.actions==6] = 0
        self.x_direction[self.actions==7] = 1 #.5

        figure()

        quiver(self.x_direction,self.y_direction)
        axis([-0.5, self.N - 0.5, -0.5, self.N - 0.5])

    def reward_curve(self, log=False, filter=1.):
        fig = figure()
        xlabel('trials')
        ylabel('total reward received')
        rewards = array(self.reward_list)
        # calculate a running average over the latencies with a averaging time 'filter'
        for i in range(1,rewards.shape[0]):
            rewards[i] = rewards[i-1] + (rewards[i] - rewards[i-1])/float(filter)

        if not log:
            plot(rewards)
        else:
            semilogy(rewards)

        fig.savefig('reward_curve_runs_%s_trials_%s_epsilon_%s.png' % (self.N_runs, self.N_trials, self.epsilon_start))


    ############################################################################
    # Private methods
    ############################################################################

    def _init_run(self):
        """
        Initialize the weights, eligibility trace, position etc.
        """

        # initialize the weights and the eligibility trace
        self.w = numpy.random.rand(self.N * self.N, 8)
        self.e = numpy.zeros((self.N * self.N, 8))

        # list that contains the times it took the agent to reach the target for all trials
        # serves to track the progress of learning
        self.latency_list = []
        self.reward_list = []

        # initialize the state and action variables
        self.x_position = 0.1
        self.y_position = 0.1
        self.action = None


    def _learn_run(self, N_trials=10):
        """
        Run a learning period consisting of N_trials trials.

        Options:
        N_trials :     Number of trials

        Note: The Q-values are not reset. Therefore, running this routine
        several times will continue the learning process. If you want to run
        a completely new simulation, call reset() before running it.

        """
        for trial in range(N_trials):
            # run a trial and store the time it takes to the target
            latency, reward = self._run_trial()
            print "%2d: Escape latency: %4d, Reward: %2d" % (trial, latency, reward)

            self.latency_list.append(latency)
            self.reward_list.append(reward)
            self.reset_pos()

            self.epsilon = sqrt(1. / (1 + trial**2))
            #self.epsilon = 0.792474 - 0.0247379 * trial + 0.000200856 * trial**2
            #self.epsilon = self.epsilon_for(trial)

        return array(self.latency_list), array(self.reward_list)


    def _run_trial(self, N_max=10000, visualize=False):
        """
        Run a single trial on the gridworld until the agent reaches the reward position.
        Return the time it takes to get there.

        Options:
        visual: If 'visualize' is 'True', show the time course of the trial graphically
        """

        # initialize the latency (time to reach the target) for this trial
        latency = 0.
        total_reward = 0.

        # start the visualization, if asked for
        if visualize:
            self._init_visualization()

        # run the trial
        self._choose_action()
        while not self._arrived() and latency < N_max:
            self._update_state()
            self._choose_action()
            total_reward += self._reward()
            self._update_weights()
            if visualize:
                self._visualize_current_state()

            latency = latency + 1


        if visualize:
            self._close_visualization()
        return latency, total_reward

    def _r(self, sx, sy, i, j):
        xj = i * (1 / 19.)
        yj = j * (1 / 19.)
        return exp( - (pow((sx - xj), 2) + pow((sy - yj), 2)) / (2 * pow(self.sigma, 2)) )


    def _Q(self, sx, sy, a):
        q = 0;
        for i in range(20):
            for j in range(20):
                q += self.w[i + 20 * j, a] * self._r(sx, sy, i, j)
        return q


    def _update_weights(self):
        """
        Update the current estimate of the Q-values according to SARSA.
        """
        # compute delta_t
        delta_t = self._reward() - \
            (self._Q(self.x_position_old, self.y_position_old, self.action_old) \
            - self.gamma * self._Q(self.x_position, self.y_position, self.action))

        # update the eligibility trace
        self.e = self.lambda_eligibility * self.gamma * self.e
        for i in range(20):
            for j in range(20):
                self.e[i + 20 * j, self.action_old] += self._r(self.x_position_old, self.y_position_old, i, j)

        # update the weights
        if self.action_old != None:
            self.w += self.eta * delta_t * self.e


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
            actions = []
            for a in range(8):
                actions.append(self._Q(self.x_position, self.y_position, a))
            self.action = argmax(actions)


    def _arrived(self):
        """
        Check if the agent has arrived.
        """

        return  pow((self.x_position - self.reward_position[0]), 2) + \
                pow((self.y_position - self.reward_position[1]), 2) < 0.01


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

        self.x_position += self.l * cos(2 * pi * self.action / 8.)
        self.y_position += self.l * sin(2 * pi * self.action / 8.)

        # check if the agent has bumped into a wall.
        if self._is_wall():
            self.x_position = self.x_position_old
            self.y_position = self.y_position_old
            self._wall_touch = True
        else:
            self._wall_touch = False


    def _is_wall(self,x_position=None,y_position=None):
        """
        This function returns, if the given position is within an obstacle
        If you want to put the obstacle somewhere else, this is what you have
        to modify. The default is a wall that starts in the middle of the room
        and ends at the right wall.

        If no position is given, the current position of the agent is evaluated.
        """
        if x_position == None or y_position == None:
            x_position = self.x_position
            y_position = self.y_position

        # check of the agent is trying to leave the gridworld
        if x_position < 0 or x_position >= 1.0 or y_position < 0 or y_position >= 1.0:
            return True

        # if none of the above is the case, this position is not a wall
        return False
