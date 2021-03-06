# Run Experiments, print results (and possibly save data) on a Bandit Problem to be specified

using HDF5
using Distributed

# DO YOU WANT TO SAVE RESULTS?
typeExp = "Save"
#typeExp = "NoSave"

# TYPE OF DISTRIBUTION
@everywhere typeDistribution="Bernoulli"

@everywhere include("EpsilonBAIalgos.jl")

# CHANGE NAME (save mode)
fname="results/TESTEps"

# BANDIT PROBLEM
@everywhere mu=[0.4 0.5 0.6 0.7 0.8]
@everywhere epsilon=0.15

@everywhere Aeps=findall(x->x>=maximum(mu)-epsilon,mu)
@everywhere Aepsilon = [Aeps[i][2] for i in 1:length(Aeps)]

# RISK LEVEL
@everywhere delta=0.1

# NUMBER OF SIMULATIONS
N=100

# POLICIES (look for their names in EpsilonBAIalgos.jl)
@everywhere policies=[TrackAndStopD,KLLUCB,UGapE,KLRacing,TaSD]
@everywhere namesPolicies=["TrackAndStop-D","KLLUCB","UGapE","KLRacing","TaS-0"]

# EXPLORATION RATES
@everywhere explo(t,delta)=log((log(t) + 1)/delta)

lP=length(policies)
rates=[explo for i in 1:lP]

# Printing optimal solution

@everywhere Tstar,optWeights=OptimalWeightsEpsilon(mu,epsilon)
@everywhere M,K = size(optWeights)
print("Theoretical number of samples: $(Tstar*log(1/delta))\n")
if (M>1)
	print("several optimal weights\n")
    for m in 1:M
        print("w_$(m) = $(optWeights[m,:])\n")
    end
else 
    print("Optimal weights: $(optWeights[1,:])\n\n")
end


# Running experiments 

function MCexp(mu,delta,N)
	for imeth=1:lP
		Draws=zeros(N,K)
		policy=policies[imeth]
		beta=rates[imeth]
		startTime=time()
		Reco,Draws = @distributed ((x,y) -> (vcat(x[1],y[1]),vcat(x[2],y[2]))) for n in 1:N
				rec,dra = policy(mu,epsilon,delta,beta)
				rec,dra
		end
		Error=collect([(r in Aepsilon) ? 0 : 1 for r in Reco])
		FracNT=sum([r==0 for r in Reco])/N
		FracReco=zeros(K)
		proportion = zeros(K)
		for k in 1:K
			FracReco[k]=sum([(r==k) ? 1 : 0 for r in Reco])/(N*(1-FracNT))
		end
		for n in 1:N
			if (Reco[n]!=0)
			    proportion += Draws[n,:]/sum(Draws[n,:])
			end
		end
		proportion = proportion / (N*(1-FracNT))
		print("Results for $(policy), average on $(N) runs\n")
		print("proportion of runs that did not terminate: $(FracNT)\n")
		print("average number of draws: $(sum(Draws)/(N*(1-FracNT)))\n")
		print("average proportions of draws: $(proportion)\n")
		print("proportion of errors: $(sum(Error)/(float(N*(1-FracNT))))\n")
		print("proportion of recommendation made when termination: $(FracReco)\n")
		print("elapsed time: $(time()-startTime)\n\n")
	end
end

# Running experiments and saving results

function SaveData(mu,delta,N)
	K=length(mu)
    for imeth=1:lP
        Draws=zeros(N,K)
        policy=policies[imeth]
		beta=rates[imeth]
        namePol=namesPolicies[imeth]
        startTime=time()
		Reco,Draws = @distributed ((x,y) -> (vcat(x[1],y[1]),vcat(x[2],y[2]))) for n in 1:N
	        reco,draws = policy(mu,epsilon,delta,beta)
	        reco,draws
	    end
		FracNT=sum([r==0 for r in Reco])/N
        FracReco=zeros(K)
		proportion = zeros(K)
        for k in 1:K
            FracReco[k]=sum([(r==k) ? 1 : 0 for r in Reco])/(N*(1-FracNT))
		end
		for n in 1:N
			if (Reco[n]!=0)
			   proportion += Draws[n,:]/sum(Draws[n,:])
			end
        end
		proportion = proportion / N
		error = 1-sum([FracReco[i] for i in Aepsilon])
        print("Results for $(policy), average on $(N) runs\n")
	    print("proportion of runs that did not terminate: $(FracNT)\n")
	    print("average number of draws: $(sum(Draws)/(N*(1-FracNT)))\n")
		print("average proportions of draws: $(proportion)\n")
	    print("proportion of errors: $(error)\n")
        print("proportion of recommendation made when termination: $(FracReco)\n")
        print("elapsed time: $(time()-startTime)\n\n")
        name="$(fname)_$(namePol)_delta_$(delta)_N_$(N).h5"
        h5write(name,"mu",mu)
        h5write(name,"delta",delta)
        h5write(name,"FracNT",collect(FracNT))
        h5write(name,"FracReco",FracReco)
        h5write(name,"Draws",Draws)
		h5write(name,"Error",error)
		h5write(name,"epsilon",epsilon)
	end
end


if (typeExp=="Save")
   SaveData(mu,delta,N)
else
   MCexp(mu,delta,N)
end
