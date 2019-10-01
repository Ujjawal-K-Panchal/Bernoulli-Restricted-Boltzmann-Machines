# Bernoulli-Restricted-Boltzmann-Machines
A self made implementation of the Bernoulli Restricted Boltzmann Machine using Contrastive Divergence & Gibbs Sampling in Python.

# Requirements : 
1. Pytorch.

# How to use?
1. Put the RBM.py file in your project's folder.
2. Import RBM like you'd import a normal Python Library. (from RBM import *)
3. An example of usage is provided here [(Movie Recommendation using Restricted Boltzmann machines)](https://github.com/Ujjawal-K-Panchal/Movie-Recommendation-using-Restricted-Boltzmann-Machines).
# Class Description : 
  # 1. rbm = RBM(nv, nh)
    nv = Number of visible nodes. (Determined by number of features in your i/p.)
    nh = Number of hidden nodes. (Number of functions to learn in the hidden node.)
  # 2. rbm.sample_h(x)
    x = Values in the visible node.
    returns : P(h|x), Bernoulli(P(h|x))
      a.P(h|x) = Probability of hidden node's activation given the input values (x) in the visible nodes.
      b.Bernoulli(P(h|x)) = Activations of the hidden neurons based on the Bernoulli Distribution.
  # 3. rbm.sample_v(x)
    x = Values in the hidden node.
    
    returns : P(v|x), Bernoulli(P(v|x))
        a.P(v|x) = Probability of visible node's activation given the values (x) in the hidden nodes.
        b.Bernoulli(P(v|x)) = Activations of the visible neurons based on 
          the Bernoulli Distribution and the acquired probability P(v|x).
  # 4. rbm.train(v0, vk, ph0, pk)
       v0 = The correct values in the initial input.
       vk = acquired values in visible nodes after k iterations of the Gibbs Sampling (G.S.).
       ph0 = Probability of hidden node's activation given the input values (v0) in the visible nodes.
       phk = Probability of hidden node's activation given the input values 
             after k iterations of G.S. (vk) in the visible nodes.
    
       returns : null.
       purpose : Used to train the weights (W), biases from visible nodes to hidden nodes (a) 
                 & biases from the hidden nodes to the visible nodes (b) 
                 of the RBM according to the Contrastive Divergence formulae given below.
   ![image of formulae](http://eric-yuan.me/wp-content/uploads/2014/07/5551.jpg)
        
 

