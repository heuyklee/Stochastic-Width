--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  The ResNet model definition
--

local nn = require 'nn'
require 'cunn'
require '../SpatialConvolution2'
require '../SpatialBatchNormalization2'

local Convolution = cudnn.SpatialConvolution
local Convolution2 = SpatialConvolution2
local Avg = cudnn.SpatialAveragePooling
local ReLU = cudnn.ReLU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization
local SBatchNorm2 = SpatialBatchNormalization2

local function createModel(opt)
   local depth = opt.depth
   local shortcutType = opt.shortcutType or 'B'
   local iChannels

   -- The shortcut layer is either identity or 1x1 convolution
   local function shortcut(nInputPlane, nOutputPlane, stride)
      local useConv = shortcutType == 'C' or
         (shortcutType == 'B' and nInputPlane ~= nOutputPlane)
      if useConv then
         -- 1x1 convolution
         return nn.Sequential()
            :add(Convolution(nInputPlane, nOutputPlane, 1, 1, stride, stride))
            :add(SBatchNorm(nOutputPlane))
      elseif nInputPlane ~= nOutputPlane then
         -- Strided, zero-padded identity shortcut
         return nn.Sequential()
            :add(nn.SpatialAveragePooling(1, 1, stride, stride))
            :add(nn.Concat(2)
               :add(nn.Identity())
               :add(nn.MulConstant(0))) -- 기존 입력에 모두 0 곱한 것을 그냥 Concat -> dim out 두배
      else
         return nn.Identity()
      end
   end

   -- The basic residual layer block for 18 and 34 layer network, and the
   -- CIFAR networks
   local function basicblock(n, stride)
      local nInputPlane = iChannels
      iChannels = n

      local s = nn.Sequential()
      s:add(Convolution2(nInputPlane,n,3,3,stride,stride,1,1))
      s:add(SBatchNorm2(n))
      s:add(ReLU(true))
      s:add(Convolution2(n,n,3,3,1,1,1,1))
      s:add(SBatchNorm2(n))

      return nn.Sequential()
         :add(nn.ConcatTable()
            :add(s)
            :add(shortcut(nInputPlane, n, stride)))
         :add(nn.CAddTable(true))
         :add(ReLU(true))
   end

   -- The bottleneck residual layer for 50, 101, and 152 layer networks
   --[[
   local function bottleneck(n, stride)
      local nInputPlane = iChannels
      iChannels = n * 4

      local s = nn.Sequential()
      s:add(Convolution(nInputPlane,n,1,1,1,1,0,0))
      s:add(SBatchNorm(n))
      s:add(ReLU(true))
      s:add(Convolution(n,n,3,3,stride,stride,1,1))
      s:add(SBatchNorm(n))
      s:add(ReLU(true))
      s:add(Convolution(n,n*4,1,1,1,1,0,0))
      s:add(SBatchNorm(n * 4))

      return nn.Sequential()
         :add(nn.ConcatTable()
            :add(s)
            :add(shortcut(nInputPlane, n * 4, stride)))
         :add(nn.CAddTable(true))
         :add(ReLU(true))
   end
   --]]

   -- Creates count residual blocks with specified number of features
   local function layer(block, features, count, stride)
      local s = nn.Sequential()
      for i=1,count do
         s:add(block(features, i == 1 and stride or 1))
      end
      return s
   end

   local model = nn.Sequential()
   -- giyobe
   -- model:initConvTbl()
   -- end giyobe

   --[[
   if opt.dataset == 'imagenet' then
      -- Configurations for ResNet:
      --  num. residual blocks, num features, residual block function
      local cfg = {
         [18]  = {{2, 2, 2, 2}, 512, basicblock},
         [34]  = {{3, 4, 6, 3}, 512, basicblock},
         [50]  = {{3, 4, 6, 3}, 2048, bottleneck},
         [101] = {{3, 4, 23, 3}, 2048, bottleneck},
         [152] = {{3, 8, 36, 3}, 2048, bottleneck},
      }

      assert(cfg[depth], 'Invalid depth: ' .. tostring(depth))
      local def, nFeatures, block = table.unpack(cfg[depth])
      iChannels = 64
      print(' | ResNet-' .. depth .. ' ImageNet')

      -- The ResNet ImageNet model
      model:add(Convolution(3,64,7,7,2,2,3,3))
      model:add(SBatchNorm(64))
      model:add(ReLU(true))
      model:add(Max(3,3,2,2,1,1))
      model:add(layer(block, 64, def[1]))
      model:add(layer(block, 128, def[2], 2))
      model:add(layer(block, 256, def[3], 2))
      model:add(layer(block, 512, def[4], 2))
      model:add(Avg(7, 7, 1, 1))
      model:add(nn.View(nFeatures):setNumInputDims(3))
      model:add(nn.Linear(nFeatures, 1000))
   -]]
   if opt.dataset == 'cifar10' then
      -- Model type specifies number of layers for CIFAR-10 model
      assert((depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110, 1202')
      local n = (depth - 2) / 6
      iChannels = 16
      print(' | ResNet-' .. depth .. ' CIFAR-10')

      -- The ResNet CIFAR-10 model
      model:add(Convolution(3,16,3,3,1,1,1,1))
      model:add(SBatchNorm(16))
      model:add(ReLU(true))
      model:add(layer(basicblock, 16, n))
      model:add(layer(basicblock, 32, n, 2))
      model:add(layer(basicblock, 64, n, 2))
      model:add(Avg(8, 8, 1, 1))
      model:add(nn.View(64):setNumInputDims(3))
      model:add(nn.Linear(64, 10))

      -- giyobe
      -- model.convTbl 만들어주기 위한 작업 수행
      -- model.convTbl에는 SpatialConvolution2 module에 대한 포인터를 가지고 있어서
      -- model.convTbl[1~18](depth:20 인 경우)와 같이 접근하여 사용
      if depth == 20 then
         -- model:initConvTbl()
         model.convTbl = {}
         model.BNTbl = {}
         tt = {{4,1,1,1,1},{4,1,1,1,4},{4,2,1,1,1},{4,2,1,1,4},{4,3,1,1,1},{4,3,1,1,4},
               {5,1,1,1,1},{5,1,1,1,4},{5,2,1,1,1},{5,2,1,1,4},{5,3,1,1,1},{5,3,1,1,4},
               {6,1,1,1,1},{6,1,1,1,4},{6,2,1,1,1},{6,2,1,1,4},{6,3,1,1,1},{6,3,1,1,4}}
         for i=1,#tt do
            model.convTbl[i] = model:get(tt[i][1]):get(tt[i][2]):get(tt[i][3]):get(tt[i][4]):get(tt[i][5])
            model.BNTbl[i] = model:get(tt[i][1]):get(tt[i][2]):get(tt[i][3]):get(tt[i][4]):get(tt[i][5]+1)
         end
         
         -- nInputPlane ~= nOutputPlane 경우에 bypassRate 0으로 set
         -- 또는 weight mat의 크기가 다른 경우
         -- 변경 금지 
         model.convTbl[1]:setBypassRate(0)
         model.convTbl[7]:setBypassRate(0)
         model.convTbl[8]:setBypassRate(0)
         model.convTbl[13]:setBypassRate(0)
         model.convTbl[14]:setBypassRate(0)
         -- 변경 금지 

         -- 우선 테스트를 위해 2,4,6 에서만 각 노드가 0.5의 확률로 bypass 되도록 설정
         -- 만일 네트워크 전체에 대한 테스트 시 아래부분 주석처리 후 SpatialConvolution2.lua
         -- 상단에 BYPASS_RATE 설정 이후 수행 가능
         --
         model.convTbl[2]:setBypassRate(0.2)
         model.convTbl[3]:setBypassRate(0.2)
         model.convTbl[4]:setBypassRate(0.2)
         model.convTbl[5]:setBypassRate(0.2)
         model.convTbl[6]:setBypassRate(0.2)
         -- model.convTbl[9]:setBypassRate(0.5)
         -- model.convTbl[10]:setBypassRate(0.5)
         -- model.convTbl[11]:setBypassRate(0.5)
         -- model.convTbl[12]:setBypassRate(0.5)
         -- model.convTbl[17]:setBypassRate(0.5)
         --
         --[[
         model.convTbl[2]:setBypassRate(0.5)
         model.convTbl[5]:setBypassRate(0.5)
         model.convTbl[9]:setBypassRate(0.5)
         model.convTbl[12]:setBypassRate(0.5)
         --]]

         -- 테스트 용도
         model.RLTbl = {}
         model.RLTbl[1] = model:get(4):get(1):get(1):get(1):get(3)
         model.RLTbl[2] = model:get(6):get(3):get(1):get(1):get(3)

      else
         print('아직은 depth 20 에만 구현하였음')
	 assert(false)
      end
      -- end giyobe

   --[[
   elseif opt.dataset == 'cifar100' then
      -- Model type specifies number of layers for CIFAR-100 model
      assert((depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110, 1202')
      local n = (depth - 2) / 6
      iChannels = 16
      print(' | ResNet-' .. depth .. ' CIFAR-100')

      -- The ResNet CIFAR-100 model
      model:add(Convolution(3,16,3,3,1,1,1,1))
      model:add(SBatchNorm(16))
      model:add(ReLU(true))
      model:add(layer(basicblock, 16, n))
      model:add(layer(basicblock, 32, n, 2))
      model:add(layer(basicblock, 64, n, 2))
      model:add(Avg(8, 8, 1, 1))
      model:add(nn.View(64):setNumInputDims(3))
      model:add(nn.Linear(64, 100))
   --]]
   else
      error('invalid dataset: ' .. opt.dataset)
   end

   local function ConvInit(name)
      for k,v in pairs(model:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:normal(0,math.sqrt(2/n))
         if cudnn.version >= 4000 then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
      end
   end
   local function BNInit(name)
      for k,v in pairs(model:findModules(name)) do
         v.weight:fill(1)
         v.bias:zero()
      end
   end

   ConvInit('cudnn.SpatialConvolution')
   ConvInit('nn.SpatialConvolution')
   BNInit('fbnn.SpatialBatchNormalization')
   BNInit('cudnn.SpatialBatchNormalization')
   BNInit('nn.SpatialBatchNormalization')
   for k,v in pairs(model:findModules('nn.Linear')) do
      v.bias:zero()
   end
   model:cuda()

   if opt.cudnn == 'deterministic' then
      model:apply(function(m)
         if m.setMode then m:setMode(1,1,1) end
      end)
   end

   model:get(1).gradInput = nil

   return model
end

return createModel
