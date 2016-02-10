require 'torch'
require 'nn'

csv2tensor = require 'csv2tensor'

--Uploading the data from the csv files into tensors
data = csv2tensor.load("train_data_short.csv", {exclude={"0", "1"}})
class = csv2tensor.load("train_news_data.csv", {include={"1"}})

--Building the network
model = nn.Sequential()
model:add(nn.Linear(3, 7))
model:add(nn.LogSoftMax())

--Building the criterion
criterion = nn.ClassNLLCriterion()
for int = 1, 3 do
	prediction = model:forward(data)
	loss = criterion:forward(prediction, class)
	model:zeroGradParameters()
	grad = criterion:backward(prediction, class)
	model:backward(data, grad)
	mlp:updateParameters(.1)
end

--Uploading the test data
testdata = csv2tensor.load("test_data_really_small.csv", {exclude={"0", "1"}})
print(tostring(model:forward(test_data)))
print(tostring("class: " .. tostring(class)))
