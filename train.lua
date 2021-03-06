--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  The training loop and learning rate schedule


local optim = require 'optim'

local M = {}
local Trainer = torch.class('resnet.Trainer', M)

function Trainer:__init(model, criterion, opt, optimState)
   self.model = model
   self.criterion = criterion
   self.optimState = optimState or {
      learningRate = opt.LR,
      learningRateDecay = 0.0,
      momentum = opt.momentum,
      nesterov = true,
      dampening = 0.0,
      weightDecay = opt.weightDecay,
   }
   self.opt = opt
   self.params, self.gradParams = model:getParameters()
   self.nConvTbl = #self.model.convTbl
   self.confMat = torch.Tensor(10,10):zero()
   --=============================================
   -- confusion matrix 만들 때 사용하는 변수
   self.CONFMAT_START_EPOCH = opt.nEpochs-5
   self.CONFMAT_END_EPOCH = opt.nEpochs
   --=============================================
   assert(self.CONFMAT_START_EPOCH <= self.CONFMAT_END_EPOCH)
end

function Trainer:train(epoch, dataloader)
   -- Trains the model for a single epoch
   self.optimState.learningRate = self:learningRate(epoch)

   local timer = torch.Timer()
   local dataTimer = torch.Timer()

   local function feval()
      return self.criterion.output, self.gradParams
   end

   local trainSize = dataloader:size()
   local top1Sum, top5Sum, lossSum = 0.0, 0.0, 0.0
   local N = 0

   -- giyobe
   -- 각 SpatialConvolution2 module에서 전체 네트워크의 SpatialConvolution2 module에 접근할
   -- 수 있도록 해당 테이블을 세팅해준다.
   -- 처음과 마지막 conv의 neighbor는 manually set
   self.model.convTbl[1]:setNeighborConv(nil, self.model.convTbl[2], 1)
   self.model.convTbl[self.nConvTbl]:setNeighborConv(self.model.convTbl[self.nConvTbl-1], nil, self.nConvTbl)
   for i = 2,self.nConvTbl-1 do
      self.model.convTbl[i]:setNeighborConv(self.model.convTbl[i-1], self.model.convTbl[i+1], i)
   end

   -- bypass 결정을 mini-batch 단위가 아닌 epoch 단위로 수행할 수 있도록 해보았다.
   -- epoch 여러개 마다 한 번씩 수행하도록...
   --[[
   for i = 1,self.nConvTbl do
      self.model.convTbl[i]:saveKernels()
      self.model.BNTbl[i]:saveKernels()
      self.model.convTbl[i]:determineBypass()
   end
   --]]
   -- end giyobe

   print('=> Training epoch # ' .. epoch)
   -- set the batch norm to training mode
   self.model:training()
   for n, sample in dataloader:run() do
      -- giyobe
      -- bypass kernel을 결정하고 이들의 weight를 로드, 저장, 0또는 1로 세팅하는 작업 수행
      for i = 1,self.nConvTbl do
         self.model.convTbl[i]:loadKernels()
         self.model.BNTbl[i]:loadKernels(self.model.convTbl[i].seltbl)

         self.model.convTbl[i]:saveKernels()
         self.model.BNTbl[i]:saveKernels()
         self.model.convTbl[i]:determineBypass()
      end
      --[[
      for i = 1,#self.model.convTbl do
	 self.model.convTbl[i]:determineBypass()
	 --self.model.convTbl[i]:loadKernels()
	 --self.model.convTbl[i]:saveKernels()
      end
      --]]
      --[[ depre: 위의 for문이 같은 역할
      for k, v in pairs(self.model:findModules('SpatialConvolution2')) do
         v:bypass() 
      end
      --]]
      -- end giyobe

      local dataTime = dataTimer:time().real

      -- Copy input and target to the GPU
      self:copyInputs(sample)

      -- giyobe
      -- forward 하기 전에 weight를 복사해서 앞으로 옮겨놓는 stochastic weight propagation
      -- 우선 중간에 발산하지 않고 약 20% 정도 까지 error가 떨어진다.
      --[[
      for i = 2,#self.model.convTbl do
         if self.model.convTbl[i].bypassRate ~= 0 then
	    for _, idx in ipairs(self.model.convTbl[i].seltbl) do
               self.model.convTbl[i].weight[idx]:copy(self.model.convTbl[i-1].weight[idx])
            end
         end 
      end
      --]]
      -- end giyobe

      -- giyobe
      -- 이 위치에서 bypass kernel(이하 BK)의 weight를 모두 0으로 만들어 준다.
      -- 이유: backward에서 gradInput을 생성하는 과정에서 우리가 임의로 만들어 넣은
      --       BK(하나의 요소만 1인)가 gradInput에 영향을 주지 않도록 하기 위해서
      --       1125_2400에 backward에서 forward 전으로 옮겼음
      for i = 2,self.nConvTbl do -- 어차피 i==1 에서 bR 무조건 0 따라서 i는 2부터
         self.model.convTbl[i]:makeBKzero()
         -- self.model.convTbl[i]:makeBRKzero() -- 1126_0040에 추가되었음
         -- conv 전 BN의 weight가 앞에 있는 conv의 gradOutput에 영향 미치므로 여기서
         -- (또는 backward 전에서도 가능)BN의 weight를 뒤로 복사해서 가지도록 한다.
         --
         for _,idx in ipairs(self.model.convTbl[i].seltbl) do
            self.model.BNTbl[i].weight[idx] = self.model.BNTbl[i-1].weight[idx]
            self.model.BNTbl[i].bias[idx] = self.model.BNTbl[i-1].bias[idx]
         end
         --
      end
      -- end giyobe

      local output = self.model:forward(self.input):float()
      local batchSize = output:size(1)
      local loss = self.criterion:forward(self.model.output, self.target)

      -- giyobe
      -- 위에서 forward전에 있던 것을 forward 뒤로 넘겨서 실험
      --[[
      for i = 2,self.nConvTbl do
         for _,idx in ipairs(self.model.convTbl[i].seltbl) do
            self.model.BNTbl[i].weight[idx] = self.model.BNTbl[i-1].weight[idx]
            self.model.BNTbl[i].bias[idx] = self.model.BNTbl[i-1].bias[idx]
         end
      end 
      --]]
      -- end giyobe

      -- giyobe
      -- 여기서는 backward 시 제대로 된 gradInput 생성을 위하여 우리가 임의로 수정했던
      -- BK 값(하나의 원소만 1)을 수정하여  이전 layer의 weight값을 복사해서 붙여준다.
      -- (즉, gradInput이 적용되기를 원하는 weight값을 가져와 그대로 사용)
      -- i = 7, 13의 경우에는 앞에서 dim, size가 변하는 conv layer이므로 bypass에서 생략
      --[[
      for i = 2,#self.model.convTbl do
         if self.model.convTbl[i].bypassRate ~= 0 then
	    for _, idx in ipairs(self.model.convTbl[i].seltbl) do
               self.model.convTbl[i].weight[idx]:copy(self.model.convTbl[i-1].weight[idx])
               self.model.convTbl[i].output:narrow(2,idx,1):copy(self.model.convTbl[i-1].output:narrow(2,idx,1))
            end
         end 
      end
      --]]
      -- end giyobe

      -- giyobe
      --[[
      print(self.model.convTbl[2].seltbl)
      for _, idx in ipairs(self.model.convTbl[2].seltbl) do
         print(idx)
         print(self.model.convTbl[2].output:narrow(2,idx,1):mean()) 
         print(self.model.convTbl[1].output:narrow(2,idx,1):mean())
         print(self.model.convTbl[2].weight[idx]:mean())
         -- self.model.convTbl[i-1].gradWeight[idx]:copy(self.model.convTbl[i].gradWeight[idx])
      end
      assert(false)
      --]]
      -- end giyobe
      
      self.model:zeroGradParameters()
      self.criterion:backward(self.model.output, self.target)
      self.model:backward(self.input, self.criterion.gradInput)

      -- giyobe
      -- backward를 통해 생성된 gradWeight를 이전 layer의 gradWeight에 더해주기 위해(BK 한정)
      -- 1122_1900 이후에 gradWeight가 아닌 gradOutput을 통한 접근으로 사용하지 않음
      for i = #self.model.convTbl,2,-1 do
         if self.model.convTbl[i].bypassRate ~= 0 then
	    for _, idx in ipairs(self.model.convTbl[i].seltbl) do
               self.model.convTbl[i-1].gradWeight[idx]:add(self.model.convTbl[i].gradWeight[idx])
               -- self.model.convTbl[i-1].gradBias[idx] = self.model.convTbl[i-1].gradBias[idx] + self.model.convTbl[i].gradBias[idx]
               -- self.model.convTbl[i-1].gradWeight[idx]:add(self.model.convTbl[i].gradWeight[idx]):mul(0.5)
               -- self.model.convTbl[i-1].gradWeight[idx]:copy(self.model.convTbl[i].gradWeight[idx])
               -- 1126_1630 BN에 대한 backprop도 고려
               self.model.BNTbl[i-1].gradWeight[idx] = self.model.BNTbl[i-1].gradWeight[idx] + self.model.BNTbl[i].gradWeight[idx]
               self.model.BNTbl[i-1].gradBias[idx] = self.model.BNTbl[i-1].gradBias[idx] + self.model.BNTbl[i].gradBias[idx]
            end
         end 
      end
      -- end giyobe

      -- giyobe
      -- gradInput normalize 해주는 부분
      -- epoch 커질 때, weight.mean 값에 비해 1이 상대적으로 훨씬 큰 값이므로
      -- gradInput 값이 이상해질 수밖에 없다. 이를 해결해보자.
      -- for k, v in pairs(self.model:findModules('SpatialConvolution2')) do
      --    v:normalizeGradInput() 
      -- end
      -- end giyobe

      -- 이 위치에서 bk의 gradOutput을 gradInput에 더해준다.
      -- 이유: 현재 bk의 gradOutput이 제대로 backprop되지 않고 있기 때문에 이를
      --       gradInput에 더해줌으로써 제대로 backprop될 수 있도록(보류)
      -- giyobe
      -- for k, v in pairs(self.model:findModules('SpatialConvolution2')) do
      --   v:gradOutput2gradInput() 
      -- end
      -- end giyobe

      optim.sgd(feval, self.params, self.optimState)

      local top1, top5 = self:computeScore(output, sample.target, 1, epoch)
      top1Sum = top1Sum + top1*batchSize
      top5Sum = top5Sum + top5*batchSize
      lossSum = lossSum + loss*batchSize
      N = N + batchSize

      print((' | Epoch: [%d][%d/%d]    Time %.3f  Data %.3f  Err %1.4f  top1 %7.3f  top5 %7.3f'):format(
         epoch, n, trainSize, timer:time().real, dataTime, loss, top1, top5))
      
      -- check that the storage didn't get changed do to an unfortunate getParameters call
      assert(self.params:storage() == self.model:parameters()[1]:storage())

      timer:reset()
      dataTimer:reset()
   end
   -- giyobe
   -- test에 들어가기 전에 kernel값을 load
   for i = 1,self.nConvTbl do
      self.model.convTbl[i]:loadKernels()
      self.model.BNTbl[i]:loadKernels(self.model.convTbl[i].seltbl)
      self.model.convTbl[i]:setNeighborConv(nil, nil, nil) -- 수행하지 않을 시 stack overflow 발생
      -- 1125_1430 추가
      self.model.convTbl[i].seltbl={}
   end
   -- end giyobe

   return top1Sum / N, top5Sum / N, lossSum / N
end

function Trainer:test(epoch, dataloader)
   -- Computes the top-1 and top-5 err on the validation set
   local timer = torch.Timer()
   local dataTimer = torch.Timer()
   local size = dataloader:size()

   local nCrops = self.opt.tenCrop and 10 or 1
   local top1Sum, top5Sum = 0.0, 0.0
   local N = 0

   self.model:evaluate()
   for n, sample in dataloader:run() do
      local dataTime = dataTimer:time().real

      -- Copy input and target to the GPU
      self:copyInputs(sample)

      local output = self.model:forward(self.input):float()
      local batchSize = output:size(1) / nCrops
      local loss = self.criterion:forward(self.model.output, self.target)

      local top1, top5 = self:computeScore(output, sample.target, nCrops, epoch)
      top1Sum = top1Sum + top1*batchSize
      top5Sum = top5Sum + top5*batchSize
      N = N + batchSize

      print((' | Test: [%d][%d/%d]    Time %.3f  Data %.3f  top1 %7.3f (%7.3f)  top5 %7.3f (%7.3f)'):format(
         epoch, n, size, timer:time().real, dataTime, top1, top1Sum / N, top5, top5Sum / N))
      
      timer:reset()
      dataTimer:reset()
   end
   self.model:training()

   print((' * Finished epoch # %d     top1: %7.3f  top5: %7.3f\n'):format(
      epoch, top1Sum / N, top5Sum / N))

   return top1Sum / N, top5Sum / N
end

-- giyobe
function Trainer:makeConfusionMatrix(pred, target, epoch)
   -- CONFMAT_START_EPOCH 부터 CONFMAT_END_EPOCH 까지 
   -- 테스트 결과를 취합해 confusion matrix를 만드는 함수
   if self.model.train == false and epoch >= self.CONFMAT_START_EPOCH and epoch <= self.CONFMAT_END_EPOCH then
      for i = 1,pred:size(1) do
         self.confMat[target[i]][pred[i][1]] = self.confMat[target[i]][pred[i][1]] + 1
      end
      if epoch == self.CONFMAT_END_EPOCH then
         self.confMat:cdiv(self.confMat:sum(2):expandAs(self.confMat))
      end
   end
end
-- end giyobe

function Trainer:computeScore(output, target, nCrops, epoch)
   if nCrops > 1 then
      -- Sum over crops
      output = output:view(output:size(1) / nCrops, nCrops, output:size(2))
         --:exp()
         :sum(2):squeeze(2)
   end

   -- Coputes the top1 and top5 error rate
   local batchSize = output:size(1)

   local _ , predictions = output:float():sort(2, true) -- descending

   -- giyobe
   self:makeConfusionMatrix(predictions, target, epoch)
   -- end giyobe

   -- Find which predictions match the target
   local correct = predictions:eq(
      target:long():view(batchSize, 1):expandAs(output))

   -- Top-1 score
   local top1 = 1.0 - (correct:narrow(2, 1, 1):sum() / batchSize)

   -- Top-5 score, if there are at least 5 classes
   local len = math.min(5, correct:size(2))
   local top5 = 1.0 - (correct:narrow(2, 1, len):sum() / batchSize)

   return top1 * 100, top5 * 100
end

function Trainer:copyInputs(sample)
   -- Copies the input to a CUDA tensor, if using 1 GPU, or to pinned memory,
   -- if using DataParallelTable. The target is always copied to a CUDA tensor
   self.input = self.input or (self.opt.nGPU == 1
      and torch.CudaTensor()
      or cutorch.createCudaHostTensor())
   self.target = self.target or (torch.CudaLongTensor and torch.CudaLongTensor()or torch.CudaTensor())
   self.input:resize(sample.input:size()):copy(sample.input)
   self.target:resize(sample.target:size()):copy(sample.target)
end

function Trainer:learningRate(epoch)
   -- Training schedule
   local decay = 0
   if self.opt.dataset == 'imagenet' then
      decay = math.floor((epoch - 1) / 30)
   elseif self.opt.dataset == 'cifar10' then
      decay = epoch >= 163 and 3 or epoch >= 122 and 2 or epoch >= 81 and 1 or 0
   elseif self.opt.dataset == 'cifar100' then
      decay = epoch >= 122 and 2 or epoch >= 81 and 1 or 0
   end
   return self.opt.LR * math.pow(0.1, decay)
end

return M.Trainer
