function [B] = Construct_Graph(X,~,anchor_rate,opts,~,isGraph)

if (~exist('opts','var'))
   opts. style = 1;
   opts. toy = 0;
   opts. IterMax = 50;
end
if nargin < 6
    isGraph = 0;
end 
IterMax = opts. IterMax;
if isGraph == 1
    B = X;
    n_view = length(X);
    [n,m] = size(X{1});
else
    k =10;
    if isfield(opts,'k')
        k = opts.k;
    end
    n_view = length(X);
    n = size(X{1},1);
    XX = [];
    for v = 1:length(X)

       XX = [XX X{v}];
    end
    m = fix(n*anchor_rate);
    B = cell(n_view,1);
    centers = cell(n_view,1);
%%
    disp('----------Anchor Selection----------');
    tic;
    if opts. style == 1 % direct sample
%         [~,ind,~] = graphgen_anchor(XX,m);
        [~,ind,~] = gen_anchor_std_en(XX,m);
        for v = 1:n_view
        centers{v} = X{v}(ind, :);
        end
    elseif opts. style == 2 % rand sample
        vec = randperm(n);
        ind = vec(1:m);
        for v = 1:n_view
            centers{v} = X{v}(ind, :);
        end
    elseif opts. style == 3
        XX = [];
        for v = 1:n_view
            XX = [XX X{v}];
        end
        [~, ~, ~, ~, dis] = litekmeans(XX, m);
        [~,ind] = min(dis,[],1);
        ind = sort(ind,'ascend');
        for v = 1:n_view
            centers{v} = X{v}(ind, :);
        end
    elseif opts. style == 4 % kmeans sample
        XX = [];
        for v = 1:n_view
           XX = [XX X{v}];
           len(v) = size(X{v},2);
        end
        [~, Cen, ~, ~, ~] = litekmeans(XX, m);
        t1 = 1;
        for v=1:n_view
           t2 = t1+len(v)-1;
           centers{v} = Cen(:,t1:t2);
           t1 = t2+1;
        end
    end
    toc;

    tic;
%%
    disp('----------Single Graphs Inilization----------');
    for v = 1:n_view
        D = L2_distance_1(X{v}', centers{v}');
        [~, idx] = sort(D, 2); % sort each row
        B{v} = zeros(n,m);
        for ii = 1:n
            id = idx(ii,1:k+1);
            di = D(ii, id);
            B{v}(ii,id) = (di(k+1)-di)/(k*di(k+1)-sum(di(1:k))+eps);
        end
    end
    toc;
end