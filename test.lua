-- nn.Concat을 이용하여 기존 conv 대체 실험한 파일
require 'nn'
require 'cudnn'
require 'Select2' -- Select를 수정하여 2D가 아닌 3D 텐서를 출력하도록 한 nn.Select2()

model = nn.Concat(1)

for i=1,2 do
   local seq = nn.Sequential()
   seq:add(nn.Concat(1):add(nn.SpatialConvolution(1,1, 3,3, 1,1, 1,1):noBias())
		       :add(nn.Select2(1,1)))
      :add(nn.Select2(1,2))

   model:add(seq)
end
model:get(1):get(2).index = 1

input = torch.Tensor():ones(1,3,3)

print(model:forward(input))

model:zeroGradParameters()

gradInput = torch.Tensor():ones(2,3,3)
gradInput = gradInput * 2

model:backward(input, gradInput)
-- model:cuda()
