function [final_result_SECGTL] = SECGTL(sample,para_p)
anchor_rate = 1;
p = para_p;
lambda = 1;
betaf = 0.6;
IterMax = 25;
X = sample{1};
Y = sample{2};
nC = length(unique(Y));
for i = 1:size(X,2)
    C = X(:,i);
    F = zeros(length(C),length(unique(C)));
    for j = 1:length(C)
        F(j,C(j)) = 1;
    end
    A = F * F.';
    sum_2 = sum(A,2);
    for j = 1:length(A)
        A(j,:) = A(j,:)/sum_2(j);
    end
    S{i} = A;
end
nV = size(X,2);

[final_result_SECGTL] = My_model_main(S, Y, nC, nV, IterMax, anchor_rate, p, lambda, betaf);