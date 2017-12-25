def trainAgent(self, Agent):

    total_steps = 0
    steps = []
    episodes = []
    for i_episode in range(20):
        state = self.env.reset()
        total_reward = 0
        while True:
            # self.env.render()

            action = Agent.trainPolicy(state)

            newState, reward, terminated, info = self.env.step(action)

            if terminated: reward = 10

            Agent.observe(state, action, reward, newState, terminated)

            if total_steps > self.memory_size:
                Agent.learn()
            '''
            if total_steps % 1000 == 0:
                print('\tTotal steps: ', total_steps)
            '''
            total_reward += reward

            if terminated:
                print('Episode ', i_episode, '\ttotal reward: ', total_reward)
                steps.append(total_steps)
                episodes.append(i_episode)
                break

            state = newState
            total_steps += 1
