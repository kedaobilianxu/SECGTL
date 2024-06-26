function [x,objV] = wshrinkObj_weight_lp(x, rho, sX, isWeight, mode, p)
if isWeight == 1
    mode = 1;
end
if ~exist('mode','var')
    C = sqrt(sX(3)*sX(2));
end
X=reshape(x,sX);

if mode == 1
    Y=X2Yi(X,3);
elseif mode == 3
    Y=shiftdim(X, 1);
else
    Y = X;
end

Yhat = dct(Y,[],3);
objV = 0;

if mode == 1
    n3 = sX(2);
elseif mode == 3
    n3 = sX(1);
else
    n3 = sX(3);
end

if isinteger(n3/2)
    endValue = int16(n3/2+1);
    for i = 1:endValue
        [uhat,shat,vhat] = svd(full(Yhat(:,:,i)),'econ');
        
        if isWeight
            weight = C./(diag(shat) + eps);
            tau = rho*weight;
            shat = soft(shat,diag(tau));
        else
            tau = rho;
            shat=diag(shat);
            shat = solve_Lp_w(shat, tau, p);
            shat=diag(shat);
        end
        
        objV = objV + sum(shat(:));
        Yhat(:,:,i) = uhat*shat*vhat';
        if i > 1
            Yhat(:,:,n3-i+2) = conj(uhat)*shat*conj(vhat)';
            objV = objV + sum(shat(:));
        end
    end
    [uhat,shat,vhat] = svd(full(Yhat(:,:,endValue+1)),'econ');
    if isWeight
        weight = C./(diag(shat) + eps);
        tau = rho*weight;
        shat = soft(shat,diag(tau));
    else
        tau = rho;
        shat=diag(shat);
        shat = solve_Lp_w(shat, tau, p);
        shat=diag(shat);
    end
    objV = objV + sum(shat(:));
    Yhat(:,:,endValue+1) = uhat*shat*vhat';
else
    endValue = int16(n3/2+1);
    for i = 1:endValue
        [uhat,shat,vhat] = svd(full(Yhat(:,:,i)),'econ');
        if isWeight
            weight = C./(diag(shat) + eps);
            tau = rho*weight;
            shat = soft(shat,diag(tau));
        else
            tau=rho;
            shat=diag(shat);
            shat = solve_Lp_w(shat, tau, p);
            shat=diag(shat);
        end
        objV = objV + sum(shat(:));
        Yhat(:,:,i) = uhat*shat*vhat';
        if i > 1
            Yhat(:,:,n3-i+2) = conj(uhat)*shat*conj(vhat)';
            objV = objV + sum(shat(:));
        end
    end
end

Y = idct(Yhat,[],3);

if mode == 1
    X = Yi2X(Y,3);
elseif mode == 3
    X = shiftdim(Y, 2);
else
    X = Y;
end

x = X(:);

end
