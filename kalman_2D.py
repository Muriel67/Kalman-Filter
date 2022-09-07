#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import math
dt = 0.1;
t = np.linspace(0,
               100,
               num = 1001)


# In[5]:


U=1; # Speed m/s
dir = -45*(math.pi)/180; # Direction of movement


# In[6]:


TrueVelx = np.zeros([1, len(t)],dtype=float);
TrueVely = np.zeros([1, len(t)],dtype=float);
Velx = np.zeros([1,len(t)],dtype=float);
Vely = np.zeros([1,len(t)],dtype=float);

Velx[:,:] = U*(math.cos(dir));
Vely[:,:] = U*(math.sin(dir));

TrueVelx[:,:] = Velx;
TrueVely[:,:] = Vely;
con=np.vstack((TrueVelx, TrueVely));
          


# In[7]:


Xinitial = [0,20];
TruePos = np.zeros([2, len(t)],dtype=float);


# In[8]:


TruePos[:,0] # returns the first columm


# In[9]:


TruePos[:,0]= Xinitial


# In[10]:


for k in np.arange(1,len(t+1),1):
    TruePos[:,k] = np.add(TruePos[:,k-1] , np.multiply(con[:,k],  0.1 ));


# In[11]:


import matplotlib.pyplot as plt
plt.title("True Position of time v/s X direction ")

plt.plot(t, TruePos[0,:])
plt.xlabel("time in (secs)")
plt.ylabel("position in X direction in (m)")
plt.grid()
plt.show()

plt.title("True Position of time v/s Y direction ")

plt.plot(t, TruePos[1,:])
plt.xlabel("time in (secs)")
plt.ylabel("position in Y direction in (m)")
plt.grid()
plt.show()

plt.title("True Position of X v/s Y direction ")

plt.plot(TruePos[0,:], TruePos[1,:])
plt.xlabel("position in X direction in (m)")
plt.ylabel("position in Y direction in (m)")
plt.grid()
plt.show()


# In[14]:


v = [1, 1];
D = np.diag(v)


# In[15]:


D


# In[16]:



R = np.power(D,2);


# In[17]:


MeasPos = np.zeros([2, len(t)],dtype=float);
MeasVel = np.zeros([2, len(t)],dtype=float);


# In[18]:


MeasPos[:,:] = np.add(TruePos[:,:],np.dot(R,np.random.rand(2,len(t))));


# In[19]:


plt.title("True and Measured Position of X v/s Y direction ")
plt.plot(MeasPos[0,:], MeasPos[1,:],'-.r')
plt.plot(TruePos[0,:], TruePos[1,:],'b')
plt.xlabel("position in X direction in (m)")
plt.ylabel("position in Y direction in (m)")
plt.legend(['Measured Position', 'True Position'])
plt.grid()
plt.show()


# In[20]:


QV = [0.05, 0.05];
QD = np.diag(QV);
Q = np.power(QD,2);
MeasVel[:,:] = np.add(con[:,:],np.dot(Q,np.random.rand(2,len(t))));


# In[21]:


plt.title("True and Measured Velocity of X v/s Y direction ")
plt.plot(t,MeasVel[0,:], t,MeasVel[1,:],'-.r')
plt.plot(t,con[0,:], t,con[1,:],'b')
plt.xlabel("velocity in X direction in (m)")
plt.ylabel("velocity in Y direction in (m)")
plt.legend(['Measured velocity', 'True velocity'])
plt.grid()
plt.show()


# In[22]:


C = np.identity(2, dtype = float);
A = np.identity(2, dtype = float);
L = np.identity(2, dtype = float);


# In[37]:


xhat = np.zeros([2,len(t)],dtype=float);
xohat = [0,0];
xhat[:,0] = xohat;
PV = [20, 20];
PD = np.diag(PV);
Pohat = np.power(PD,2);
P = np.zeros([2,2],dtype=float);
H = np.zeros([2,2],dtype=float);
P = Pohat;
print(P)


# In[44]:


for k in np.arange(1,len(t+1),1):
    xhat[:,k] = np.add(xhat[:,k-1] , np.multiply(MeasVel[:,k],  0.1 ));
    P = np.add(np.dot(np.dot(A,P),np.transpose(A)), np.dot(np.dot(L,Q),np.transpose(L)))
    P = np.subtract(P,np.dot(P,np.dot(C,np.dot(np.linalg.inv(np.add(np.dot(np.dot(C,P),np.transpose(C)),R)),np.dot(C,P)))))
    H = np.dot(P,np.dot(np.transpose(C),np.linalg.inv(R)))
    xhat[:,k] = np.add(xhat[:,k],np.dot(H,np.subtract(MeasPos[:,k],np.dot(C,xhat[:,k]))))


# In[47]:


plt.plot(xhat[0,:],xhat[1,:],'-.m')
plt.plot(MeasPos[0,:], MeasPos[1,:],'-.r')
plt.plot(TruePos[0,:], TruePos[1,:],'b')

plt.xlabel("Estimated Position in X direction in (m)")
plt.ylabel("Estimated Position in Y direction in (m)")
plt.legend(['Estimated Position using Kalman Filter','Measured Position contains noise' ,'True position'])
plt.grid()
plt.show()


# In[ ]:





# In[ ]:




