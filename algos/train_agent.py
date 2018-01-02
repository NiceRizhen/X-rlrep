def trainAgent(self, Agent):

    total_steps = 0
    history = []
    for i_episode in range(1000):
        observation = self.env.reset()
        total_reward = 0
        for step in range(2000):
            # self.env.render()

            action = Agent.trainPolicy(observation)

            observation_, reward, done, info = self.env.step(action)

            if done: reward = 10

            Agent.observe(observation, action, reward, observation_, done)

            if total_steps > self.memory_size:
                Agent.learn()
            '''
            if total_steps % 1000 == 0:
                print('\tTotal steps: ', total_steps)
            '''
            total_reward += reward
            if done:
                break

            observation = observation_
            total_steps += 1

        print("Episode #%d \tReward %d" % (i_episode, total_reward))
        history.append(total_reward)

    return history
