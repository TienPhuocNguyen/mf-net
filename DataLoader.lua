require 'image'

local M = {}

local DataLoader = torch.class('DataLoader', M)

function DataLoader:__init(dir)
  self.dir = dir
  self.cnt = 1
end

function DataLoader:read()
  local fname = (self.dir .. '/in%06d' .. '.jpg'):format(self.cnt)
  self.cnt = self.cnt + 1

  local img = image.load(fname, 1, 'byte')

  return img
end


return M.DataLoader
