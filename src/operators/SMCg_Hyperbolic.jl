@inline function sinh_env(x::T,y::T,z::T) where {T<:AbstractFloat}
  return sinh(y)-sinh(x)-(y-x)*cosh(x)
end
@inline function sinh_envd(x::T,y::T,z::T) where {T<:AbstractFloat}
  return (y-x)*sinh(x)
end
@inline function cv_sinh(x::T,xL::T,xU::T) where {T<:AbstractFloat}
  p::T = zero(T)
  if (xL>=zero(T))
    return sinh(x),cosh(x)
  elseif (xU<=zero(T))
    return line_seg(x,xL,sinh(xL),xU,sinh(xU)),dline_seg(x,xL,sinh(xL),xU,sinh(xU),cosh(x))
  else
    try
      p = newton(xL,xL,zero(T),sinh_env,sinh_envd,xU,zero(T))
    catch e
      if isa(e, ErrorException)
        p = golden_section(xL,zero(T),sinh_env,xU,zero(T))
      end
    end
    if (x>p)
      return sinh(x),cosh(x)
    else
      return line_seg(x,p,sinh(p),xU,sinh(xU)),dline_seg(x,p,sinh(p),xU,sinh(xU),cosh(x))
    end
  end
end
@inline function cc_sinh(x::T,xL::T,xU::T) where {T<:AbstractFloat}
  p::T = zero(T)
  if (xL>=zero(T))
    return line_seg(x,xL,sinh(xL),xU,sinh(xU)),dline_seg(x,xL,sinh(xL),xU,sinh(xU),cosh(x))
  elseif (xU<=zero(T))
    return sinh(x),cosh(x)
  else
    try
      p = newton(xU,zero(T),xU,sinh_env,sinh_envd,xL,zero(T))
    catch e
      if isa(e, ErrorException)
        p = golden_section(zero(T),xU,sinh_env,xL,zero(T))
      end
    end
    if (x>p)
      return line_seg(x,xL,sinh(xL),p,sinh(p)),dline_seg(x,xL,sinh(xL),p,sinh(p),cosh(x))
    else
      return sinh(x),cosh(x)
    end
  end
end
@inline function sinh(x::SMCg{N,T}) where {N,T<:AbstractFloat}
  eps_max::T = x.Intv.hi
  eps_min::T = x.Intv.lo
  midcc::T,cc_id::Int64 = mid3(x.cc,x.cv,eps_max)
  midcv::T,cv_id::Int64 = mid3(x.cc,x.cv,eps_min)
  cc::T,dcc::T = cc_sinh(midcc,x.Intv.lo,x.Intv.hi)
  cv::T,dcv::T = cv_sinh(midcv,x.Intv.lo,x.Intv.hi)
  if (MC_param.mu >= 1)
    gcc1::T,gdcc1::T = cc_sinh(x.cv,x.Intv.lo,x.Intv.hi,c)
    gcv1::T,gdcv1::T = cv_sinh(x.cv,x.Intv.lo,x.Intv.hi,c)
    gcc2::T,gdcc2::T = cc_sinh(x.cc,x.Intv.lo,x.Intv.hi,c)
    gcv2::T,gdcv2::T = cv_sinh(x.cc,x.Intv.lo,x.Intv.hi,c)
    cv_grad::SVector{N,T} = max(zero(T),gdcv1)*x.cv_grad + min(zero(T),gdcv2)*x.cc_grad
    cc_grad::SVector{N,T} = min(zero(T),gdcc1)*x.cv_grad + max(zero(T),gdcc2)*x.cc_grad
  else
    cc_grad = mid_grad(x.cc_grad, x.cv_grad, cc_id)*dcc
    cv_grad = mid_grad(x.cc_grad, x.cv_grad, cv_id)*dcv
  end
  return SMCg{N,T}(cc, cv, cc_grad, cv_grad, sinh(x.Intv),x.cnst, x.IntvBox, x.xref)
end

@inline function asinh_env(x::T,y::T,z::T) where {T<:AbstractFloat}
  return (asinh(z)-asinh(x))/(z-x)-one(T)/sqrt(one(T)+x^2)
end
@inline function asinh_envd(x::T,y::T,z::T) where {T<:AbstractFloat}
  return (asinh(z)-asinh(x))/(z-x)^2+x/(x^2+one(T))^(3/2)-one(T)/((z-x)*sqrt(x^2+one(T)))
end
@inline function cv_asinh(x::T,xL::T,xU::T) where {T<:AbstractFloat}
  p::T = zero(T)
  if (xL>=zero(T))
    return line_seg(x,xL,asinh(xL),xU,asinh(xU)),dline_seg(x,xL,asinh(xL),xU,asinh(xU),one(T)/sqrt(x^2+one(T)))
  elseif (xU<=zero(T))
    return asinh(x),one(T)/sqrt(x^2+one(T))
  else
    try
      p = newton(xL/2,xL,zero(x),asinh_env,asinh_envd,xL,xU)
    catch e
      if isa(e, ErrorException)
        p = golden_section(xL,zero(x),asinh_env,xL,xU)
      end
    end
    if (x<=p)
      return asinh(x),one(T)/sqrt(x^2+one(T))
    else
      return line_seg(x,p,asinh(p),xU,asinh(xU)),dline_seg(x,p,asinh(p),xU,asinh(xU),one(T)/sqrt(x^2+one(T)))
    end
  end
end
@inline function cc_asinh(x::T,xL::T,xU::T) where {T<:AbstractFloat}
  p = zero(T)
  if (xL>=zero(T))
    return asinh(x),one(T)/sqrt(x^2+one(T))
  elseif (xU<=zero(T))
    return line_seg(x,xL,asinh(xL),xU,asinh(xU)),dline_seg(x,xL,asinh(xL),xU,asinh(xU),one(T)/sqrt(x^2+one(T)))
  else
    try
      p = newton(xU/2,zero(x),xU,asinh_env,asinh_envd,xL,xU)
    catch e
      if isa(e, ErrorException)
        p = golden_section(zero(x),xU,asinh_env,xL,xU)
      end
    end
    if (x<=p)
      return line_seg(x,xL,asinh(xL),p,asinh(p)),dline_seg(x,xL,asinh(xL),p,asinh(p),one(T)/sqrt(x^2+one(T)))
    else
      return asinh(x),one(T)/sqrt(x^2+one(T))
    end
  end
end
@inline function asinh(x::SMCg{N,T}) where {N,T<:AbstractFloat}
  eps_max::T = x.Intv.hi
  eps_min::T = x.Intv.lo
  midcc::T,cc_id::Int64 = mid3(x.cc,x.cv,eps_max)
  midcv::T,cv_id::Int64 = mid3(x.cc,x.cv,eps_min)
  cc::T,dcc::T = cc_asinh(midcc,x.Intv.lo,x.Intv.hi)
  cv::T,dcv::T = cv_asinh(midcv,x.Intv.lo,x.Intv.hi)
  if (MC_param.mu >= 1)
    gcc1::T,gdcc1::T = cc_asinh(x.cv,x.Intv.lo,x.Intv.hi,c)
    gcv1::T,gdcv1::T = cv_asinh(x.cv,x.Intv.lo,x.Intv.hi,c)
    gcc2::T,gdcc2::T = cc_asinh(x.cc,x.Intv.lo,x.Intv.hi,c)
    gcv2::T,gdcv2::T = cv_asinh(x.cc,x.Intv.lo,x.Intv.hi,c)
    cv_grad::SVector{N,T} = max(zero(T),gdcv1)*x.cv_grad + min(zero(T),gdcv2)*x.cc_grad
    cc_grad::SVector{N,T} = min(zero(T),gdcc1)*x.cv_grad + max(zero(T),gdcc2)*x.cc_grad
  else
    cc_grad = mid_grad(x.cc_grad, x.cv_grad, cc_id)*dcc
    cv_grad = mid_grad(x.cc_grad, x.cv_grad, cv_id)*dcv
  end
  return SMCg{N,T}(cc, cv, cc_grad, cv_grad, asinh(x.Intv),x.cnst, x.IntvBox, x.xref)
end

@inline function tanh_env(x::T,y::T,z::T) where {T<:AbstractFloat}
  return (tanh(y)-tanh(x))/(one(T)-tanh(x)^2)-y+x
end
@inline function tanh_envd(x::T,y::T,z::T) where {T<:AbstractFloat}
  return -((one(T)-tanh(x)^2)^(-2))*(tanh(y)-tanh(x))
end
@inline function cv_tanh(x::T,xL::T,xU::T) where {T<:AbstractFloat}
  p::T = zero(T)
  if (xL>=zero(T))
    return line_seg(x,xL,tanh(xL),xU,tanh(xU)),dline_seg(x,xL,tanh(xL),xU,tanh(xU),sech(x)^2)
  elseif (xU<=zero(T))
    return tanh(x),sech(x)^2
  else
    try
      p = newton(xL,xL,zero(T),tanh_env,tanh_envd,xU,zero(T))
    catch e
      if isa(e, ErrorException)
        p = golden_section(xL,zero(T),tanh_env,xU,zero(T))
      end
    end
    if (x<=p)
      return tanh(x),sech(x)^2
    else
      return line_seg(x,p,tanh(p),xU,tanh(xU)),dline_seg(x,p,tanh(p),xU,tanh(xU),sech(x)^2)
    end
  end
end
@inline function cc_tanh(x::T,xL::T,xU::T) where {T<:AbstractFloat}
  p::T = zero(T)
  if (xL>=zero(T))
    return tanh(x),sech(x)^2
  elseif (xU<=zero(T))
    return line_seg(x,xL,tanh(xL),xU,tanh(xU)),dline_seg(x,xL,tanh(xL),xU,tanh(xU),sech(x)^2)
  else
    try
      p = newton(xU/2,zero(T),xU,tanh_env,tanh_envd,xL,xU)
    catch e
      if isa(e, ErrorException)
        p = golden_section(zero(T),xU,tanh_env,xL,zero(T))
      end
    end
    if (x<=p)
      return line_seg(x,xL,tanh(xL),p,tanh(p)),dline_seg(x,xL,tanh(xL),p,tanh(p),sech(x)^2)
    else
      return tanh(x),sech(x)^2
    end
  end
end
@inline function tanh(x::SMCg{N,T}) where {N,T<:AbstractFloat}
  eps_max::T = x.Intv.hi
  eps_min::T = x.Intv.lo
  midcc::T,cc_id::Int64 = mid3(x.cc,x.cv,eps_max)
  midcv::T,cv_id::Int64 = mid3(x.cc,x.cv,eps_min)
  cc::T,dcc::T = cc_tanh(midcc,x.Intv.lo,x.Intv.hi)
  cv::T,dcv::T = cv_tanh(midcv,x.Intv.lo,x.Intv.hi)
  if (MC_param.mu >= 1)
    gcc1::T,gdcc1::T = cc_tanh(x.cv,x.Intv.lo,x.Intv.hi,c)
    gcv1::T,gdcv1::T = cv_tanh(x.cv,x.Intv.lo,x.Intv.hi,c)
    gcc2::T,gdcc2::T = cc_tanh(x.cc,x.Intv.lo,x.Intv.hi,c)
    gcv2::T,gdcv2::T = cv_tanh(x.cc,x.Intv.lo,x.Intv.hi,c)
    cv_grad::SVector{N,T} = max(zero(T),gdcv1)*x.cv_grad + min(zero(T),gdcv2)*x.cc_grad
    cc_grad::SVector{N,T} = min(zero(T),gdcc1)*x.cv_grad + max(zero(T),gdcc2)*x.cc_grad
  else
    cc_grad = mid_grad(x.cc_grad, x.cv_grad, cc_id)*dcc
    cv_grad = mid_grad(x.cc_grad, x.cv_grad, cv_id)*dcv
  end
  return SMCg{N,T}(cc, cv, cc_grad, cv_grad, tanh(x.Intv),x.cnst, x.IntvBox, x.xref)
end

@inline function atanh_env(x::T,y::T,z::T) where {T<:AbstractFloat}
  return (x-y)-(one(T)-x^2)*(atan(x)-atan(y))
end
@inline function atanh_envd(x::T,y::T,z::T) where {T<:AbstractFloat}
  return one(T)+2*x*(atan(x)-atan(y))
end
@inline function cv_atanh(x::T,xL::T,xU::T) where {T<:AbstractFloat}
  p::T = zero(T)
  if (xL>=zero(T))
    return atanh(x),one(T)/(one(T)-x^2)
  elseif (xU<=zero(T))
    return line_seg(x,xL,atanh(xL),xU,atanh(xU)),dline_seg(x,xL,atanh(xL),xU,atanh(xU),one(T)/(one(T)-x^2))
  else
    try
      p = newton(xU,zero(T),xU,atanh_env,atanh_envd,xL,xU)
    catch e
      if isa(e, ErrorException)
        p = golden_section(xL,zero(T),atanh_env,xU,zero(T))
      end
    end
    if (x>p)
      return atanh(x),one(T)/(one(T)-x^2)
    else
      return line_seg(x,p,atanh(p),xU,atanh(xU)),dline_seg(x,p,atanh(p),xU,atanh(xU),one(T)/(one(T)-x^2))
    end
  end
end
@inline function cc_atanh(x::T,xL::T,xU::T) where {T<:AbstractFloat}
  p::T = zero(T)
  if (xL>=zero(T))
    return line_seg(x,xL,atanh(xL),xU,atanh(xU)),dline_seg(x,xL,atanh(xL),xU,atanh(xU),one(T)/(one(T)-x^2))
  elseif (xU<=zero(T))
    return atanh(x),one(T)/(one(T)-x^2)
  else
    try
      p = newton(xL,xL,zero(T),atanh_env,atanh_envd,xL,xU)
    catch e
      if isa(e, ErrorException)
        p = golden_section(zero(T),xU,atanh_env,xL,zero(T))
      end
    end
    if (x>p)
      return line_seg(x,xU,atanh(xU),p,atanh(p)),dline_seg(x,xU,atanh(xU),p,atanh(p),one(T)/(one(T)-x^2))
    else
      return atanh(x),one(T)/(one(T)-x^2)
    end
  end
end
@inline function atanh(x::SMCg{N,T}) where {N,T<:AbstractFloat}
  eps_max::T = x.Intv.hi
  eps_min::T = x.Intv.lo
  midcc::T,cc_id::Int64 = mid3(x.cc,x.cv,eps_max)
  midcv::T,cv_id::Int64 = mid3(x.cc,x.cv,eps_min)
  cc::T,dcc::T = cc_atanh(midcc,x.Intv.lo,x.Intv.hi)
  cv::T,dcv::T = cv_atanh(midcv,x.Intv.lo,x.Intv.hi)
  if (MC_param.mu >= 1)
    gcc1::T,gdcc1::T = cc_atanh(x.cv,x.Intv.lo,x.Intv.hi,c)
    gcv1::T,gdcv1::T = cv_atanh(x.cv,x.Intv.lo,x.Intv.hi,c)
    gcc2::T,gdcc2::T = cc_atanh(x.cc,x.Intv.lo,x.Intv.hi,c)
    gcv2::T,gdcv2::T = cv_atanh(x.cc,x.Intv.lo,x.Intv.hi,c)
    cv_grad::SVector{N,T} = max(zero(T),gdcv1)*x.cv_grad + min(zero(T),gdcv2)*x.cc_grad
    cc_grad::SVector{N,T} = min(zero(T),gdcc1)*x.cv_grad + max(zero(T),gdcc2)*x.cc_grad
  else
    cc_grad = mid_grad(x.cc_grad, x.cv_grad, cc_id)*dcc
    cv_grad = mid_grad(x.cc_grad, x.cv_grad, cv_id)*dcv
  end
  return SMCg{N,T}(cc, cv, cc_grad, cv_grad, atanh(x.Intv),x.cnst, x.IntvBox, x.xref)
end
