class Identify:
    def __init__(self, start_frame):
        self.start_frame = start_frame
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.er_states = None
        self.er_actions = None
        self.er_dones = None
        self.er_rewards = None
        self.er_next_states = None

        self.Qvals = []

        self.batch_idxs = []
        self.batch_Qvals = []
        self.batch_new_Qvals = []
        self.batch_target_states_vals = []
        self.batch_target_actions = []
        self.batch_loss = []

        self.churn = []