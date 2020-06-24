function Weight = SGD(weight, input, correct_output)

alpha = 0.9;
N = 4; 

for k =1:N
    transposed_input = input(k,:)';
    d = correct_output(k);
weighted_sum = Weight*transposed_input;
output = Sigmoid(weighted_sum);

error = d - output
delta = output*(1-output)*error;

dWeight = alpha*delta*transposed_input;

Weight(1) = Weight(1) + dWeight(1);
Weight(2) = Weight(2) + dWeight(2);


end
end

