require 'cutorch'
require 'image'

local BGSubtractor = require 'BGSubtractor'
local DataLoader   = require 'DataLoader'

cmd = torch.CmdLine()
cmd:option('-n', 'net/BS_net.t7', 'directory to trained model')
cmd:option('-s', 'datasets/CDNet2014/dataset/dynamicBackground/fall/input', 'data directory')
cmd:option('-d', true, 'display')

opt = cmd:parse(arg)

local data = DataLoader(opt.s)
local alg = BGSubtractor(data:read(), opt.n)

local cnt = 0
local img

if opt.d == true then
  disp = require 'display'
end

while true do

  cnt = cnt + 1
  img = data:read()

  -- if img == nil then
  --   break
  -- end

  -- add trigger
  alg:operator(img, (cnt <= 100))

  local fg = alg:getFG()

  -- display the results
  if opt.d == true then
    disp.image(img, {win=2, title='Input'})
    disp.image(fg, {win=1, title='Foreground'})
  end

  -- image.save(("datasets/CDNet2014/results/dynamicBackground/fall/%d.png"):format(cnt),fg)

end
