import run_environment 

# Mountain car
# mc = run_environment.Playground('MountainCar-v0', [10,30], [[-1.2,.6],[-.07,.07]],.3,.1,.05,.99)
# mc.begin_training(1000)

# Cartpole
cp = run_environment.Playground('CartPole-v1', [3,10,30,30], [[-4,8,4.8],[-5,5],[-.5,.5],[-5,5]],.3,.1,.05,.99)
cp.begin_training(10000)

# Acrobot
# acro = run_environment.Playground('Acrobot-v1', [3,3,3,3,10,10], [[-1,1],[-1,1],[-1,1],[-1,1],[-12.57,12.57],[-28.28,28.28]],.3,.1,.05,.99)
# acro.begin_training(10000)

# Pendulum
# pen = run_environment.Playground('Pendulum-v0', [10,30,30], [[-1,1],[-1,1],[-8,8]],.3,.1,.05,.99)
# pen.begin_training(1000)