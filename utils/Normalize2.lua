-- this code is taken from https://github.com/jzbontar/mc-cnn. Thank to the author.
local Normalize2, parent = torch.class('nn.Normalize2', 'nn.Module')

function Normalize2:__init()
   parent.__init(self)
   self.norm = torch.CudaTensor()
end

function Normalize2:updateOutput(input)
   self.norm:resize(input:size(1), 1, input:size(3), input:size(4))
   self.output:resizeAs(input)
   cutils.Normalize_forward(input, self.norm, self.output)
   return self.output
end
