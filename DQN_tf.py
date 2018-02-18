import numpy as np
import gym
import random
import tensorflow as tf
#env=  gym.make('MountainCar-v0') 
#env = gym.make('Acrobot-v1') # change if desired to required environment
env=gym.make('CartPole-v1') #change num_actions to 2
num_actions=env.action_space.n #number of available actions
num_epochs=1000
observations_dim=np.shape(env.observation_space.high)[0] #the observations in the environment
gamma=0.99 #discount factor
visualize_after_steps=10 #start the display
memorylen=100000 #memory size of experience replay
batch_size=32 #the batch size used for experience replay
steps=np.zeros(num_epochs)
batch=np.empty([1,observations_dim])
emax=1
emin=0.05
epsilon=1  #epsilon will exponentially decrease from 1 to emin.
dparam=0.001 #decay parameter
tnuf=150#target network update frequency
num_hidden_1=64
num_hidden_2=64

np.random.seed(1234)

def translate(value, leftMin, leftMax, rightMin, rightMax): #this scales the state variables appropriately, called by normalize function
   
    leftrange = leftMax - leftMin
    rightrange = rightMax - rightMin
    #Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / leftrange
     #Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightrange)   
    
def normalize(state):
    
    normstate=np.empty(np.shape(state))
    if num_actions==2:
        val=np.array([2.5, 3.6, 0.28, 3.7]) #in case you want to set manually the limits, like in cartpole
        val1=-val
    else:
        val=env.observation_space.low
        val1=env.observation_space.high
    
    for i in range(np.shape(state)[0]):
        normstate[i]=translate(state[i],val[i],val1[i],0,1) #making into zero one range
    normstate=state
    return normstate
    

x=tf.placeholder('float32',[None,observations_dim]) #input state
y=tf.placeholder('float32',[None,num_actions]) #ground truth q values


w1=tf.Variable(tf.random_normal([observations_dim,num_hidden_1]))
bias1=tf.Variable(tf.random_normal([num_hidden_1]))
mul=tf.add(tf.matmul(x,w1),bias1)
act1=tf.nn.relu(mul)
w2=tf.Variable(tf.random_normal([num_hidden_1,num_hidden_2]))
bias2=tf.Variable(tf.random_normal([num_hidden_2]))
act2=tf.nn.relu(tf.add(tf.matmul(act1,w2),bias2))
w3=tf.Variable(tf.random_normal([num_hidden_2,num_actions]))
bias3=tf.Variable(tf.random_normal([num_actions]))
qval=tf.add(tf.matmul(act2,w3),bias3) #the outpuot q value.

#target network, which has similar architecture, but different weights.
w1_tar=tf.Variable(tf.random_normal([observations_dim,num_hidden_1]))
bias1_tar=tf.Variable(tf.random_normal([num_hidden_1]))
act1_tar=tf.nn.relu(tf.add(tf.matmul(x,w1_tar),bias1_tar))
w2_tar=tf.Variable(tf.random_normal([num_hidden_1,num_hidden_2]))
bias2_tar=tf.Variable(tf.random_normal([num_hidden_2]))
act2_tar=tf.nn.relu(tf.add(tf.matmul(act1,w2_tar),bias2))
w3_tar=tf.Variable(tf.random_normal([num_hidden_2,num_actions]))
bias3_tar=tf.Variable(tf.random_normal([num_actions]))
qval_tar=tf.add(tf.matmul(act2_tar,w3_tar),bias3_tar)

loss=tf.reduce_mean(tf.square(qval-y)) #mean squared loss
opt=tf.train.AdamOptimizer().minimize(loss) #optimizer

sess=tf.InteractiveSession() #run session
init_op=tf.initialize_all_variables()
sess.run(init_op)

def updatetarget():
    sess.run([ w1_tar.assign(w1),w2_tar.assign(w2),bias1_tar.assign(bias1),bias2_tar.assign(bias2),w3_tar.assign(w3),\
            bias3_tar.assign(bias3)]) #the copies the weight of actor network to the target network, every t steps.
    
updatetarget()   
    
def pick_action(curstate): #pick the action epsilon greedily.
    
    
    qvalues =sess.run(qval,feed_dict={x:curstate.reshape(1,observations_dim)})
    if (np.random.rand() < epsilon): #random action
        action = np.random.randint(0,num_actions)
    else: #choose greedily 
        action = (np.argmax(qvalues))
    return action
    
def compute_target(curstate): #qvalue computation by target network.
    qvalues = sess.run(qval_tar,{x:curstate.reshape(-1,observations_dim)})
    return qvalues
    
    
def updateweights(replay): #experience replay steps.
    
    batch_len = min(batch_size,len(replay))
    minibatch=random.sample(replay,batch_len)
    
    no_state = np.zeros(observations_dim)

    states = np.array([ d[0] for d in minibatch ])
    snext = np.array([ (no_state if d[2] is None else d[2]) for d in minibatch ])

    predict_cur = sess.run(qval,{x:states.reshape(-1,observations_dim)})
    predict_next = sess.run(qval,{x:snext.reshape(-1,observations_dim)})

    x_ = np.zeros((batch_len, observations_dim))
    y_ = np.zeros((batch_len, num_actions))
    
    for k in range(batch_len):
        d = minibatch[k]
        state = d[0]
        action = d[1]
        reward = d[3]
        done=d[4]
        
        m = predict_cur[k]
        if done:
            m[action] = reward
        else:
            m[action] = reward + gamma* np.amax(predict_next[k])

        x_[k] = state
        y_[k] = m

    sess.run(opt,feed_dict={x:x_,y:y_})
    

#model.summary()

replay=[]
for idx,i in enumerate(range(int(num_epochs))):
    
    curstate = env.reset()
    csn=normalize(curstate) #Normalized state values fed to the network
    while(1):
        steps[idx]+=1
        #env.render()
        curaction=pick_action(csn)
        nextstate,reward, done, info = env.step(curaction) #environment simulator called
        #print(curstate)
        nsn=normalize(nextstate)
        
        
        replay.append((csn,curaction,nsn,reward,done))
        
        if (len(replay) > memorylen): 
            replay.pop(0) #memory has been filled, start refill
            
        model=updateweights(replay) #updating the weights by calling experience replay function.
        
        if done or steps[idx]>2000:
            print('Episode %d ended in %d steps' %(idx,steps[idx]))
            break
        

        curstate=nextstate
        csn=nsn
        
        if steps[idx]%tnuf==0: #update every 25 steps
            
            epsilon = emin + (emax - emin)*np.exp(-dparam* np.sum(steps)) #decaying the exploration parameter epsilon.
            updatetarget()
     
env.monitor.close()


