-- ConvolutionSF.lua
-- Single Featuremap으로 구성된 Convolution module을 만들기 위한 코드
-- ResNet코드에서 /models/resnet.lua 중 basicblock function에서 호출되어 사용될 예정

require 'nn'
require 'cudnn'
require '../Select2'

local SFSpatialConvolution, parent = torch.class('nn.SFSpatialConvolution', 
						 'nn.SpatialConvolution')
local Conv = cudnn.SpatialConvolution
local Sel = nn.Select2

-- ConvSF에서 정해야할 것:
---- 해당 Conv layer의 deathRate
---- input, output channel의 개수
---- kernel size
---- stride
---- padding size

function SFSpatialConvolution:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH, dR)
   self.reset = function() end
   parent.__init(self, nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)

   self.dR = dR

   model = nn.Concat(1)
   for i=1,nOutputPlane do
      local seq = nn.Sequential()
      seq:add(nn.Concat(1):add(Conv(nInputPlane,1, kW,kH, dW,dH, padW,padH):noBias())
                          :add(Sel(1,1))) -- i는 제약 존재 시 그대로 i, 제약 없을 시 랜덤(중복X)
         :add(Sel(1,i)) -- 여기의 i는 1과2 중에서 랜덤으로
      model:add(seq)
   end

   self.modules = {self.model}
end

function SFSpatialConvolution:get(index)
   return self.modules[index]
end
--[[
function createConvSF(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
   return Conv(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH) 
end
--]]
