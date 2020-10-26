# Print a summary of the results + display the empirical distribution of the stopping time for several algorithms
# The data has to be stored in the results/ folder

using PyPlot
using HDF5
using Distributions

# EPSILON BAI?
EpsilonBAI ="False"

# NAME AND POLICIES NAMES (should match that of saved data)
fname="results/TEST"
#fname="results/TESTEps"
names=["TrackAndStop","ChernoffBC","UGapE","ChernoffRacing","ChernoffT3C"]
#names=["TrackAndStop-D","KLLUCB","UGapE","KLRacing","TaS-0"]
# PARAMETERS
delta=0.1
N=100

clf()

# BINS THAT CAN BE CUSTOMIZED 
xdim=length(names)
ydim=1

NBins=30
xmax=2000


xtxt=0.6*xmax
Bins=round.(Int,range(1,stop=xmax,length=NBins))

mu=h5read("$(fname)_$(names[1])_delta_$(delta)_N_$(N).h5","mu")
K=length(mu)
if EpsilonBAI == "True"
   epsilon=h5read("$(fname)_$(names[1])_delta_$(delta)_N_$(N).h5","epsilon")
end

clf()
title("mu = $(mu)")

for j in 1:length(names)
   name="$(fname)_$(names[j])_delta_$(delta)_N_$(N).h5"
   FracNT=h5read(name,"FracNT")
   Draws=h5read(name,"Draws")
   Error=h5read(name,"Error")
   FracReco=h5read(name,"FracReco")
   subplot(xdim,ydim,j)
   NbDraws=sum(Draws,dims=2)'
   proportion=zeros(N,K)
   for k in 1:N
      proportion[k,:]=Draws[k,:]/sum(Draws[k,:])
   end
   prop=mean(proportion,dims=1)
   MeanDraws=mean(NbDraws)
   StdDraws=std(NbDraws)
   histo=plt.hist(vec(NbDraws),Bins)
   Mhisto=maximum(histo[1])
   PyPlot.axis([0,xmax,0,Mhisto])
   ytxt1=0.75*Mhisto
   ytxt2=0.6*Mhisto
   ytxt3=0.5*Mhisto
   ytxt4=0.4*Mhisto
   EmpError=round.(Int,10000*Error)/10000
   axvline(MeanDraws,color="black",linewidth=2.5)
   PyPlot.text(xtxt,ytxt1,"mean = $(round(Int,MeanDraws)) (std=$(round(Int,StdDraws)))")
   PyPlot.text(xtxt,ytxt2,"delta = $(delta)")
   PyPlot.text(xtxt,ytxt3,"emp. error = $(EmpError)")
   PyPlot.text(xtxt,ytxt4,"emp. recommendation made = $(FracReco)")
   if (j==1)
      if (EpsilonBAI=="True")
         title("mu=$(mu), epsilon=$(epsilon), $(names[j])")
      else
         title("mu=$(mu), $(names[j])")
      end
   else
      title("$(names[j])")
   end
   print("Results for $(names[j]), average on $(N) runs\n")
   print("proportion of runs that did not terminate: $(FracNT)\n")
   print("average number of draws: $(MeanDraws)\n")
   print("average proportion of draws: \n $(prop)\n")
   print("proportion of errors: $(EmpError)\n")
   print("proportion of recommendation made when termination: $(FracReco)\n\n")
end
