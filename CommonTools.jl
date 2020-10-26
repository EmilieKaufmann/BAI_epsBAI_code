using Distributions

function randmax(vector,rank=1)
   # returns an integer, not a CartesianIndex
   vector=vec(vector)
   Sorted=sort(vector,rev=true)
   m=Sorted[rank]
   Ind=findall(x->x==m,vector)
   index=Ind[floor(Int,length(Ind)*rand())+1]
   return (index)
end


function dicoSolve(f, xMin, xMax, pre=1e-11)
  # find m such that f(m)=0 using binary search
  l = xMin
  u = xMax
  sgn = f(xMin)
  while (u-l>pre)
    m = (u+l)/2
    if (f(m)*sgn>0)
      l = m
    else
      u = m
    end
  end
  return (u+l)/2
end
