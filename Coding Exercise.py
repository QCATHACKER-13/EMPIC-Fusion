import numpy as np


# define constants and variables

k = 1 # (N/m) constant for Biot-Savart Law calculation 

dx, dy, dz =.25*np.pi,.25*np.pi,.0437896 #[um] distance between electrodes on each axis 


def calculate_B(x):
    """Calculates magnetic field vector using Biot Savart law"""
    
    r= x[None,:] - X[:,:]   #(num of particles by num of points)*3 array where [:,:,:] creates three subarrays with dimensions [numofparticles][numpoints][] representing point coordinates at different positions 
    rmagsqrd = sum((r)**2,-1)#calculate square magnitudes along all axes
    invrmagsqrd = 1./rmagsqrd   
    cosphi = invrmagsqrd * ((z[:,:, None]*y[:, :,None]- z[:,None, :] * y[:,:,None])*(r[...,0]+ r[...,1])/rmagsqrd**2 + (-z[:,:, None]**2+ k **(-2))*invrmagsqrd )     
    sinphi = np.sqrt(abs(cosphi))    

    return [(dy/(dz**2)*(sinphi*((z[:,:, None]*y[:, :,None]- z[:,None, : ] * y[:,:,None])+ r[...,2]/rmagsqrd)-cos(theta)), dx / (dz ** 2) *(sinphi *((z[:,:, None]*x[:, :,None]- z[:,None, : ] * x[:,:,None])+ r[...,2]/rmagsqrd), cos(theta)/(k))]  


X = [[-.2*.25*np.pi],[0]]         #{array([[[-.2]],[[0.]]]), array([[-0.        ],
       [-0.        ]], dtype='<f8')}            |       {array([[-0., -0.],
           [0., 0.]])}                             }



Y=[[-.2+.25*np.pi],
   [.2+.25*np.pi]]                [{array([[-.25],[-.25]]), array([[.25],[.25]])}]




       Z={array([[(i/float(len(Z)))*2.*np.pi -.2*np.pi],
             [(j/float(len(Z)))*2.*np.pi -.2*np.pi]]))|
            for i in range(int(.5/.2))+range((- int(.5/.2)+1),(int(.5/.2))))
        for j in range(int(.5/.2))+range((- int(.5/.2)+1),(int(.5/.2))))
        
    



   theta={(array([(k/float(len(X)))*2.*np.pi ])}|for l in len(X)):
  {(l/float(len(Y)))*2.*np.pi })
