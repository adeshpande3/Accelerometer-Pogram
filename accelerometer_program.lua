-----------------------------------------------------------------------------------------
-- This program utilizes accelerometer data (specifically the x, y, and z readings)
-- to make a classification of the current state of the user.

-- There are 3 different features that will be used as the inputs to the neural 
-- network and the output will be a value that represents a classification of what 
-- is currently happening. 

-- Input features: X reading on accelerometer, Y reading on accelerometer, Z reading on accelerometer

-- Output: One of 7 classes
--	(1) Working at Computer
--	(2) Standing Up, Walking, and Going Up/Down Stairs
--	(3) Standing
--	(4) Walking
--	(5) Going Up/Down Stairs
--	(6) Walking and Talking with Someone
--	(7) Talking while Standing

-- General Approach: For predicting a classification of the current state using the 3 features of
-- an accelerometer, we use a logistic regression model to calculate the probabilities of 
-- each of the classes.

-- Future Work: 

-- Dataset: https://archive.ics.uci.edu/ml/datasets/Activity+Recognition+from+Single+Chest-Mounted+Accelerometer
-----------------------------------------------------------------------------------------
require 'torch'
require 'nn'
csv2tensor = require 'csv2tensor'

-----------------------------------------------------------------------------------------
-- 1. Data Formatting

-- The data used by this program is placed into a csv file. 

data = csv2tensor.load("train_data_short_test3.csv", {exclude={"0"}})

-----------------------------------------------------------------------------------------
-- 2. Neural Network Architecture

model = nn.Sequential()
model:add(nn.Linear(3, 7))
model:add(nn.LogSoftMax())
criterion = 

-----------------------------------------------------------------------------------------
-- 3. Evaluation Function

x, dl_dx = model:getParameters()

feval = function()
	_nidx_ = (_nidx_ or 0) + 1
	if _nidx_ > (#train_data)[1] then _nidx_ = 1 end

	example = data[_nidx_]
	target = example[{ {1} }]
	inputs = example[{ {2, 4} }]
	dl_dx:zero()
	loss = criterion:forward(model:forward(inputs), target)
	model:backward(inputs, criterion:backward(model.output, target))
	return loss, dl_dx
end

-----------------------------------------------------------------------------------------
-- 4. Setting Hyperparameters

sgd_params = {
	learningRate = 1e-6
}

-----------------------------------------------------------------------------------------
-- 5. Training 

for i = 1,1e2 do
	current_loss = 0
	for i = 1, (#data)[1] do
		_,fs = optim.sgd(feval, x, sgd_params)
		current_loss = current_loss + fs[1]
	end
	current_loss = current_loss / (#data)[1]
	print(current_loss)
end

-----------------------------------------------------------------------------------------
-- 6. Testing

testdata = csv2tensor.load("test_data_really_small.csv", {exclude={"0", "1"}})
print(tostring(model:forward(test_data)))
print(tostring("class: " .. tostring(class)))
