# Algorithms for epsilon-Best Arm Identification in Exponential Family Bandit Models in the Fixed-Confidence Setting
# compatible with version 0.7

# The nature of the distribution should be precised by choosing a value for typeDistribution before including the current file

# All the algorithms take the following input
# mu : vector of arms means
# epsilon : value of the slack allowed for finding the best arm
# delta : risk level
# rate : the exploration rate (a function)

using Distributions
include("CommonTools.jl")
include("KLfunctions.jl")

if (typeDistribution=="Bernoulli")
   d=dBernoulli
   dup=dupBernoulli
   dlow=dlowBernoulli
   variance=VBernoulli
   muinf=0
   function sample(mu)
       (rand()<mu)
   end
   function bdot(theta)
        exp(theta)/(1+exp(theta))
   end
   function bdotinv(mu)
        log(mu/(1-mu))
   end
elseif (typeDistribution=="Poisson")
   d=dPoisson
   dup=dupPoisson
   dlow=dlowPoisson
   variance=VPoisson
   muinf = 0
   function sample(mu)
       rand(Poisson(mu))
   end
   function bdot(theta)
        exp(theta)
   end
   function bdotinv(mu)
        log(mu)
   end
elseif (typeDistribution=="Exponential")
   d=dExpo
   dup=dupExpo
   dlow=dlowExpo
   variance=VExpo
   muinf=0
   function sample(mu)
       -mu*log(rand())
   end
   function bdot(theta)
        -log(-theta)
   end
   function bdotinv(mu)
        -exp(-mu)
   end
elseif (typeDistribution=="Gaussian")
   # sigma (std) must be defined !
   d=dGaussian
   dup=dupGaussian
   dlow=dlowGaussian
   variance=VGaussian
   muinf=-Inf
   function sample(mu)
       mu+sigma*randn()
   end
   function bdot(theta)
        sigma^2*theta
   end
   function bdotinv(mu)
        mu/sigma^2
   end
end

# Define the right lambdaX (minimizer) function depending on epsilon and on the distributions

if (typeDistribution=="Gaussian")
   function lambdaX(x,mua,mub,epsilon,pre=10e-12)
      # computes the minimizer for lambda in (mu^- ; mu^+ - epsilon) of d(mua,lambda)+d(mub,lambda+epsilon) 
      # has be be used when mua > mub-epsilon !!
      return (mua + x*(mub-epsilon))/(1+x)
   end
elseif (typeDistribution=="Bernoulli")
   function lambdaX(x,mua,mub,epsilon,pre=10e-12)
      # computes the minimizer for lambda in (mu^- ; mu^+ - epsilon) of d(mua,lambda)+d(mub,lambda+epsilon) 
      # has be be used when mua > mub-epsilon !!
      if (epsilon==0)
         return (mua + x*mub)/(1+x)
      elseif (x==0)
         return mua
      else 
         #func(lambda)=(lambda-mua)/variance(lambda)+x*(lambda+epsilon-mub)/variance(lambda+epsilon)
         func(lambda)=(lambda-mua)*variance(lambda+epsilon)+x*(lambda+epsilon-mub)*variance(lambda)
         return dicoSolve(func, max(mub-epsilon,pre),min(mua,1-epsilon),pre)
      end
   end 
else 
   # Poisson or Exponential distribution
   function lambdaX(x,mua,mub,epsilon,pre=10e-12)
      # computes the minimizer for lambda in (mu^- ; mu^+ - epsilon) of d(mua,lambda)+d(mub,lambda+epsilon) 
      # has be be used when mua > mub-epsilon !!
      if (epsilon==0)
         return (mua + x*mub)/(1+x) 
      elseif (x==0)
         return mua
      else 
         #func(lambda)=(lambda-mua)/variance(lambda)+x*(lambda+epsilon-mub)/variance(lambda+epsilon)
         func(lambda)=(lambda-mua)*variance(lambda+epsilon)+x*(lambda+epsilon-mub)*variance(lambda)
         return dicoSolve(func, max(mub-epsilon,pre),mua,pre)
      end
   end 
end

# Define the right gb function as well 

if typeDistribution=="Bernoulli"
   function gb(x,mua,mub,epsilon,pre=1e-12)
      # compute the minimum value of d(mua,lambda)+d(mub,lambda+epsilon)
      # requires mua > mub - epsilon
      if (x==0)
         return d(mua,min(mua,1-epsilon))
      else 
         # works when mua=mub=1 as d(1,1)=0
         lambda = lambdaX(x,mua,mub,epsilon,pre)
         return d(mua,lambda)+x*d(mub,lambda+epsilon)
      end
   end
else
   function gb(x,mua,mub,epsilon,pre=1e-12)
      # compute the minimum value of d(mua,lambda)+d(mub,lambda+epsilon)
      # requires mua > mub - epsilon
      if (x==0)
         return 0
      else 
         lambda = lambdaX(x,mua,mub,epsilon,pre)
         return d(mua,lambda)+x*d(mub,lambda+epsilon)
      end
   end
end

# Define the function that gives the support on which to look for ystar

if typeDistribution=="Bernoulli"
   function AdmissibleAux(mu,a,epsilon)
      return d(mu[a],min(mu[a],1-epsilon)),d(mu[a],max(0,maximum([mu[i] for i in 1:(length(mu)) if i!=a].-epsilon)))
   end
elseif typeDistribution=="Gaussian"
   function AdmissibleAux(mu,a,epsilon)
      return 0,d(mu[a],maximum([mu[i] for i in 1:(length(mu)) if i!=a].-epsilon))
   end
else
   # Poisson and Exponential
   function AdmissibleAux(mu,a,epsilon)
      return 0,d(mu[a],max(0,maximum([mu[i] for i in 1:(length(mu)) if i!=a].-epsilon)))
   end
end


# COMPUTING THE OPTIMAL WEIGHTS BASED ON THE FUNCTIONS G AND LAMBDA

function AdmissibleYBern(mua,mub,epsilon)
   return d(mua,min(mua,1-epsilon)),d(mua,max(mub-epsilon,0))
end

function xbofy(y,mua,mub,epsilon,pre = 1e-12)
   # return x_b(y), i.e. finds x such that g_b(x)=y
   # requires mua > mub - epsilon
   # requires [d(mua,min(mua,muplus-epsilon)) < y < d(mua,max(mb-epsilon,muminus))]
   # CANNOT WORK when mua=mub=1 in the Bernoulli case (as the function xb is not defined)
   function g(x) 
      return gb(x,mua,mub,epsilon) - y
   end 
   xMax=1
   while g(xMax)<0
      xMax=2*xMax
   end
   xMin=0
   return dicoSolve(x->g(x), xMin, xMax,pre)
end

function auxEps(y,mu,a,epsilon,pre=1e-12)
   # returns F_mu(y) - 1
   # requires a to be epsilon optimal!
   # y has to satisfy d(mua,min(mua,muplus-epsilon)) < y < d(mua,max(max_{b\neq a} mub - epsilon,mumin))
   # (the function AdmissibleAux computes this support)
   K = length(mu)
   Indices = collect(1:K)
   deleteat!(Indices,a)
   x = [xbofy(y,mu[a],mu[b],epsilon,pre) for b in Indices]
   m = [lambdaX(x[k],mu[a], mu[Indices[k]], epsilon,pre) for k in 1:(K-1)]
   return (sum([d(mu[a],m[k])/(d(mu[Indices[k]], m[k]+epsilon)) for k in 1:(K-1)])-1)
end

function aOpt(mu,a,epsilon, pre = 1e-12)
   # returns the optimal weights and values associated for the epsilon optimal arm a
   # a has to be epsilon-optimal!
   # cannot work in the Bernoulli case if mua=1 and there is another arm with mub=1
   K=length(mu)
   yMin,yMax=AdmissibleAux(mu,a,epsilon)
   fun(y) = auxEps(y,mu,a,epsilon,pre)
   if yMax==Inf
      yMax=1
      while fun(yMax)<0
         yMax=yMax*2
      end
   end
   ystar = dicoSolve(fun, yMin, yMax, pre)
   x = zeros(1,K)
   for k in 1:K
      if (k==a)
         x[k]=1
      else
         x[k]=xbofy(ystar,mu[a],mu[k],epsilon,pre)
      end
   end
   nuOpt = x/sum(x)
   return nuOpt[a]*ystar, nuOpt
end

function OptimalWeightsEpsilon(mu,epsilon,pre=1e-11)
   # returns T*(mu) and a matrix containing as lines the candidate optimal weights
   K=length(mu)
   # find the epsilon optimal arms
   IndEps=findall(x->x>=maximum(mu)-epsilon, mu)
   L=length(IndEps)
   if (L>1)&&(epsilon==0)
      # multiple optimal arms when epsilon=0
      vOpt=zeros(1,K)
      vOpt[IndEps].=1/L
      return Inf,vOpt
   elseif (L>1)&&(maximum(mu)==1)&&(typeDistribution=="Bernoulli")
      # more than 1 maxima equal to 1 in the Bernoulli case 
      vOpt=zeros(1,K)
      Weights = zeros(L,K)
      for l in 1:L
         Weights[l,IndEps[l][2]]=1
      end
      return 1/d(1,1-epsilon),Weights
   else
      Values=zeros(1,L)
      Weights = zeros(L,K)
      for i in 1:L
         dval,weights=aOpt(mu,IndEps[i][2],epsilon,pre)
         Values[i]=1/dval
         Weights[i,:]=weights
      end
      # look at the argmin of the characteristic times
      Tchar = minimum(Values)
      iFmu=findall(x->x==Tchar, Values)
      M=length(iFmu)
      WeightsFinal = zeros(M,K)
      for i in 1:M 
         WeightsFinal[i,:]=Weights[iFmu[i][2],:]
      end
      return Tchar,WeightsFinal
   end
end

function PGLRT(muhat,counts,epsilon,Aeps,K) 
   # compute the parallel GLRT stopping rule and return the Best arm 
   # counts have to be all positive 
   Aepsilon = [Aeps[i][2] for i in 1:length(Aeps)]
   L = length(Aepsilon)
   Zvalues = zeros(Float64,1,L)
   for i in 1:L
      a = Aepsilon[i]
      NA = counts[a]
      MuA = muhat[a]
      Zvalues[i]=minimum([NA*gb(counts[b]/NA,MuA,muhat[b],epsilon) for b in 1:K if b!=a])
   end
   # pick an argmin
   Ind = argmax(Zvalues)[1]
   Best = Aepsilon[Ind]
   return maximum(Zvalues),Best
end


## ALGORITHMS 

# epsilon - Track and Stop [Garivier and Kaufmann, 2020]

function TrackAndStopD(mu,epsilon,delta,rate)
   # Chernoff stopping + D-Tracking 
   condition = true
   K=length(mu)
   N = zeros(1,K)
   S = zeros(1,K)
   # initialization
   for a in 1:K
      N[a]=1
      S[a]=sample(mu[a])
   end
   t=K
   Best=1
   while (condition)
      Mu=S./N
      # Empirical best arm
      IndMax=findall(x -> x==maximum(Mu),Mu)
      I=1
      # compute the stopping statistic
      Score,Best=PGLRT(Mu,N,epsilon,IndMax,K)
      if (Score > rate(t,delta))
         # stop
         condition=false
      elseif (t >10000000)
         # stop and outputs (0,0)
         condition=false
         Best=0
         print(N)
         print(S)
         N=zeros(1,K)
      else
         if (minimum(N) <= max(sqrt(t) - K/2,0))
            # forced exploration
            I=argmin(N)
         else
            # continue and sample an arm
            val,Weights=OptimalWeightsEpsilon(Mu,epsilon,1e-11)
            # if ties, always pick the first weight in the list 
            Dist = Weights[1,:]
            # choice of the arm
            I=argmax(Dist'-N/t)
         end
      end
      # draw the arm
      t+=1
      S[I]+=sample(mu[I])
      N[I]+=1
   end
   recommendation=Best
   return (recommendation,N)
end


function TrackAndStopC(mu,epsilon,delta,rate)
   # Chernoff stopping + C-Tracking 
   condition = true
   K=length(mu)
   N = zeros(1,K)
   S = zeros(1,K)
   # initialization
   for a in 1:K
      N[a]=1
      S[a]=sample(mu[a])
   end
   t=K
   Best=1
   SumWeights=ones(1,K)/K
   while (condition)
      Mu=S./N
      # Empirical best arm
      IndMax=findall(x -> x==maximum(Mu),Mu)
      I=1
      # compute the stopping statistic
      Score,Best=PGLRT(Mu,N,epsilon,IndMax,K)
      if (Score > rate(t,delta))
         # stop
         condition=false
      elseif (t >10000000)
         # stop and outputs (0,0)
         condition=false
         Best=0
         print(N)
         print(S)
         N=zeros(1,K)
      else
         # continue and sample an arm
         val,Weights=OptimalWeightsEpsilon(Mu,epsilon,1e-11)
         # if ties, always pick the first weight in the list 
         Dist = Weights[1,:]'
         SumWeights=SumWeights+Dist
         if (minimum(N) <= max(sqrt(t) - K/2,0))
            # forced exploration
            I=argmin(N)
         else
            I=argmax(SumWeights-N)
         end
      end
      # draw the arm
      t+=1
      S[I]+=sample(mu[I])
      N[I]+=1
   end
   recommendation=Best
   return (recommendation,N)
end

function TaSD(mu,epsilon,delta,rate)
   # Track-and-Stop with epsilon=0
   return TrackAndStopD(mu,0,delta,rate)
end

function TaSC(mu,epsilon,delta,rate)
   # Track-and-Stop with epsilon=0
   return TrackAndStopC(mu,0,delta,rate)
end
   
## CONFIDENCE BASED ALGORITHMS

# KL-LUCB [Kaufmann and Kalyanakrishnan, 2012]

function KLLUCB(mu,epsilon,delta,rate)
   condition = true
   K=length(mu)
   N = zeros(1,K)
   S = zeros(1,K)
   # initialization
   for a in 1:K
      N[a]=1
      S[a]=sample(mu[a])
   end
   t=K
   Best=1
   while (condition)
      Mu=S./N
      # Empirical best arm
      Best=randmax(Mu)
      # Find the challenger
      UCB=zeros(1,K)
      LCB=dlow(Mu[Best],rate(t,delta)/N[Best])
      for a in 1:K
	      if a!=Best
	         UCB[a]=dup(Mu[a],rate(t,delta)/N[a])
         end
      end
      Challenger=randmax(UCB)
      # draw both arms
      t=t+2
      S[Best]+=sample(mu[Best])
      N[Best]+=1
      S[Challenger]+=sample(mu[Challenger])
      N[Challenger]+=1
      # check stopping condition
      condition=(LCB < UCB[Challenger]-epsilon)
      if (t>1000000)
	      condition=false
         Best=0
         N=zeros(1,K)
      end
   end
   recommendation=Best
   return (recommendation,N)
end

# UGapE [Gabillon et al., 2012]
 
function UGapE(mu,epsilon,delta,rate)
   condition = true
   K=length(mu)
   N = zeros(1,K)
   S = zeros(1,K)
   # initialization
   for a in 1:K
      N[a]=1
      S[a]=sample(mu[a])
   end
   t=K
   Best=1
   while (condition)
      Mu=S./N
      # Empirical best arm
      Best=randmax(Mu)
      # Find the challenger
      UCB=zeros(1,K)
      LCB=zeros(1,K)
      for a in 1:K
         UCB[a]=dup(Mu[a],rate(t,delta)/N[a])
         LCB[a]=dlow(Mu[a],rate(t,delta)/N[a])
      end
      B=zeros(1,K)
      for a in 1:K
         Index=collect(1:K)
         deleteat!(Index,a)
         B[a] = maximum(UCB[Index])-LCB[a]
      end
      Value=minimum(B)
      Best=argmin(B)
      UCB[Best]=0
      Challenger=argmax(UCB)
      # choose which arm to draw
      t=t+1
      I=(N[Best]<N[Challenger]) ? Best : Challenger
      S[I]+=sample(mu[I])
      N[I]+=1
      # check stopping condition
      condition=(Value > epsilon)
      if (t>1000000)
         condition=false
         Best=0
         N=zeros(1,K)
      end
   end
   recommendation=Best[2]
   return (recommendation,N)
end

## ALGORITHM BASED ON ELIMINATIONS 

# KL-Racing [Kaufmann and Kalyanakrishnan 2013]  
 
function KLRacing(mu,epsilon,delta,rate)
   condition = true
   K=length(mu)
   N = zeros(1,K)
   S = zeros(1,K)
   # initialization
   for a in 1:K
      N[a]=1
      S[a]=sample(mu[a])
   end
   round=1
   t=K
   Remaining=collect(1:K)
   while (length(Remaining)>1)
      # Drawn all remaining arms
      for a in Remaining
         S[a]+=sample(mu[a])
         N[a]+=1
      end
      round+=1
      t+=length(Remaining)
      # Check whether the worst should be removed
      Mu=S./N
      MuR=Mu[Remaining]
      MuBest=maximum(MuR)
      Best=randmax(MuR)
      Best=Remaining[Best]
      MuWorst=minimum(MuR)
      IndWorst=randmax(-MuR)
      if (dlow(MuBest,rate(round,delta)/round) > dup(MuWorst,rate(round,delta)/round)-epsilon)
         # remove Worst arm
         deleteat!(Remaining,IndWorst)
      end
      if (t>1000000)
         Remaining=[0]
         N=zeros(1,K)
      end
   end
   recommendation=Remaining[1]
   return (recommendation,N)
end
 
