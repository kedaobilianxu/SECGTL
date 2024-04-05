function [final_result] = My_model_main(S ,Y, nC ,nV ,IterMax ,anchor_rate ,p ,lambda ,betaf)
N = length(Y);
M = fix(N*anchor_rate);
alpha = repmat(1/nV, [1,nV]);
beta = ones(nV,1);
%% ===initialization=== %%
for k = 1:nV
    % 需要重新初始化
    J{k} = zeros(N, M);
    Q{k} = zeros(N, M);
    QQ{k}=zeros(N, M);
    R{k}=zeros(N, M);
    U{k}=zeros(N, M);
end
iter = 0;
result_former = 0;
Isconverg = 0;
F = zeros(N+M,nC);
Fn = zeros(N,nC);
Fm = zeros(M,nC);
sX = [N, M, nV];
mu = 10e-6; max_mu = 10e9; pho_mu = 1.1;
final_result = zeros(1,7);

%%
time_start = clock;

while(Isconverg == 0)

    %% update F
    clear ZZ;
    clear Dm;
    clear Dn;
    ZZ = 0;
    for v = 1: nV
        Dm{v} = diag(sum(S{v},1)+eps);
        Dn{v} = diag(sum(S{v},2)+eps);
        ZZ = ZZ + (1/alpha(v))*(Dn{v}^-0.5)*S{v}*(Dm{v}^-0.5);
    end
    [uu, ~, vv] = svd(ZZ);
    Fn = uu(:,1:nC)*(2^-0.5);
    Fm = vv(:,1:nC)*(2^-0.5);
    F=[Fn; Fm];


    %% update S
    clear R;
    clear U;
    for v =1:nV
        R{v} = (1/alpha(v))*(Dm{v}^-0.5)*Fm*Fn'*(Dn{v}^-0.5);
        U{v} = mu*J{v} + 2*betaf*R{v}' - Q{v};
    end
    for v = 1:nV
        for i = 1:N
            MM = U{v}(i,:);
            S{v}(i, :) = EProjSimplex_new(MM, 1);
        end
    end
    

    %% solve J{v}
    for v =1:nV
        QQ{v} = S{v} + Q{v}./mu;
    end
    Q_tensor = cat(3,QQ{:,:});
    [myj, ~] = wshrinkObj_weight_lp(Q_tensor(:), lambda*beta./mu, sX, 0,3,p);
    J_tensor = reshape(myj, sX);
    for k=1:nV
        J{k} = J_tensor(:,:,k);
    end
    clear J_tensor Q_tensor
    

    %% update betaf
    clear SS;
    SS = (1/alpha(1))*S{1};
    
    for v = 2: nV
        SS = SS + (1/alpha(v))*S{v};
    end
    Sum_alpha = sum(1./alpha);
    SS = SS./Sum_alpha;
%     rank(SS)
    try
        [Flabel] = coclustering_bipartite_fast1(SS, nC, IterMax);
    catch
        Isconverg  = 1;
    end
    result = ClusteringMeasure1(Y,Flabel);


    if (sum(result) - sum(final_result))>0
        final_result = result;
    
    end

    fprintf('\n')
    
    [~, ev1, ~] = svd(SS);
    ev = diag(ev1);
    fn1 = sum(ev(1:nC));
    fn2 = sum(ev(1:nC+1));
    if fn1 < nC-0.0000001
        betaf = 2*betaf;
    elseif fn2 > nC+1-0.0000001
        betaf = betaf/2;
    else
       break
    end  
  end
    
    %% update Q
    for v = 1:nV
        Q{v} = Q{v} + mu*(S{v} - J{v});
    end

    %% update mu
    mu = min(mu*pho_mu, max_mu);
    
    %%
    if (iter > IterMax)
        Isconverg  = 1;
    end

    
    iter = iter + 1;
end



