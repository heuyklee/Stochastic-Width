-- 이 파일은 특정 conv feature map element를 0으로 만들면 
-- 학습 시 어떤 일이 일어나는지를 확인하기 위한 테스트 파일이었음
require 'nn'

model = nn.Sequential()
model:add(nn.SpatialConvolution(2,2, 3,3, 1,1, 1,1):noBias())
model:add(nn.SpatialConvolution(2,2, 3,3, 1,1, 1,1):noBias())
model:add(nn.SpatialConvolution(2,2, 3,3):noBias())

-- model:zeroGradParameters()
params, gradParams = model:getParameters()
params:zero()
model:get(2).weight[1][1][2][2] = 1

input = torch.Tensor():rand(1,2,3,3)
gradOutput = torch.Tensor():rand(1,2,1,1)

model:forward(input)
model:backward(input, gradOutput)
