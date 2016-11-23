local Select2, parent = torch.class('nn.Select2', 'nn.Module')

function Select2:__init(dimension,index)
   parent.__init(self)
   self.dimension = dimension
   self.index = index 
end

function Select2:updateOutput(input)
   -- giyobe
   -- input으로 128,16,32,32 feature map이 들어왔다.
   -- output으로 128,1,32,32 feature map을 출력하자.

   local sizes = input:size()

   local dim = self.dimension < 0 and input:dim() + self.dimension + 1 or self.dimension
   local index = self.index < 0 and input:size(dim) + self.index + 1 or self.index

   local output = torch.CudaTensor():resize(sizes[1], 1, sizes[3], sizes[4]):zero()

-- print(input:type())
-- assert(false)
   for i=1,sizes[1] do
      output[i][1]:copy(input[i]:select(1, index))
   end
   local sizes2 = output:size()

   self.output = torch.CudaTensor():resize(sizes2)
   -- self.output = torch.CudaTensor():resizeAs(output)
   return self.output:copy(output)
--[[
   if sizes:size() == 3 then
      local output = input:select(dim, index):resize(1, sizes[2], sizes[3])
   elseif sizes:size() == 4 then
      local output = input:select(dim, index):resize(sizes[1], 1, sizes[3], sizes[4])
   end
   self.output:resizeAs(output)
   return self.output:copy(output)
--]]
end

function Select2:updateGradInput(input, gradOutput)
   local dim = self.dimension < 0 and input:dim() + self.dimension + 1 or self.dimension
   local index = self.index < 0 and input:size(dim) + self.index + 1 or self.index
   self.gradInput:resizeAs(input)  
   self.gradInput:zero()
   self.gradInput:select(dim,index):copy(gradOutput) 
   return self.gradInput
end 
