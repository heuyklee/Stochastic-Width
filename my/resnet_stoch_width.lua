--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  The ResNet model definition

local nn = require 'nn'
require 'cunn'
require '../Select2'

local Convolution = cudnn.SpatialConvolution
local Avg = cudnn.SpatialAveragePooling
local ReLU = cudnn.ReLU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization
local Sel = nn.Select2
local Sap = nn.SpatialAveragePooling 

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
               :add(nn.MulConstant(0)))
      else
         return nn.Identity()
      end
   end

   -- The basic residual layer block for 18 and 34 layer network, and the
   -- CIFAR networks
--[[
    local function basicblock(n, stride)
      local nInputPlane = iChannels
      iChannels = n

      local s = nn.Sequential()
      s:add(Convolution(nInputPlane,n,3,3,stride,stride,1,1))
      s:add(SBatchNorm(n))
      s:add(ReLU(true))
      s:add(Convolution(n,n,3,3,1,1,1,1))
      s:add(SBatchNorm(n))

      return nn.Sequential()
         :add(nn.ConcatTable()
            :add(s)
            :add(shortcut(nInputPlane, n, stride)))
         :add(nn.CAddTable(true))
         :add(ReLU(true))
   end
--]]
   local function basicblock(n, stride)
     local nInputPlane = iChannels
     iChannels = n

     local s = nn.Sequential()
     local c = nn.Concat(2)
     for i=1,n do
        c:add(Convolution(nInputPlane,1,3,3,stride,stride,1,1))
     end
     s:add(c)
     s:add(SBatchNorm(n))
     s:add(ReLU(true))

     local c = nn.Concat(2)
     for i=1,n do
        c:add(Convolution(n,1,3,3,1,1,1,1))
     end
     s:add(c)
     s:add(SBatchNorm(n))

     return nn.Sequential()
        :add(nn.ConcatTable()
           :add(s)
           :add(shortcut(nInputPlane, n, stride)))
        :add(nn.CAddTable(true))
        :add(ReLU(true))
  end
--[[
  local function basicblock(n, stride)
      local nInputPlane = iChannels
      iChannels = n

      local s = nn.Sequential()
      local c = nn.Concat(2)
      local se = nn.Sequential()

      if stride == 1 then
         se:add(nn.Concat(2):add(Convolution(nInputPlane,1, 3,3, 1,1, 1,1))
                            :add(Sel(2,1))) -- i는 제약 존재 시 그대로 i, 제약 없을 시 랜덤(중복X)
           :add(Sel(2,1)) -- 여기의 i는 1과2 중에서 랜덤으로
      else
         se:add(nn.Concat(2):add(Convolution(nInputPlane,1, 3,3, stride,stride, 1,1))
                            :add(nn.Sequential():add(Sel(2,1))
                                               :add(nn.SpatialAveragePooling(1,1, stride,stride))))
           :add(Sel(2,1)) -- 여기의 i는 1과2 중에서 랜덤으로
      end
      for i=1,n do
         c:add(se)
      end
      s:add(c)
      s:add(SBatchNorm(n))
      s:add(ReLU(true))
      local c = nn.Concat(2)
      local se = nn.Sequential()
      se:add(nn.Concat(2):add(Convolution(n,1, 3,3, 1,1, 1,1))
                         :add(Sel(2,1))) -- i는 제약 존재 시 그대로 i, 제약 없을 시 랜덤(중복X)
        :add(Sel(2,1)) -- 여기의 i는 1과2 중에서 랜덤으로
      for i=1,n do
         c:add(se)
      end
      s:add(c)
      s:add(SBatchNorm(n))

      return nn.Sequential()
         :add(nn.ConcatTable()
            :add(s)
            :add(shortcut(nInputPlane, n, stride)))
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
