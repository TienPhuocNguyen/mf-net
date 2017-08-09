require 'nn'
require 'cudnn'
require 'cutorch'

require 'utils/libcutils'
include('utils/Normalize2.lua')

local M = {}
local util = paths.dofile('utils/util.lua')
local BGSubtractor = torch.class('BGSubtractor', M)


local GHOSTDET_D_MAX = 0.01
local GHOSTDET_S_MIN = 0.995

local FB_R_VAR = 0.01

local FB_V_INCR = 1
local FB_V_DECR = 0.1

local FB_T_INCR = 0.5
local FB_T_DECR = 0.25
local FB_T_LOWER = 2
local FB_T_UPPER = 256

local UNSTABLE_REG_RATIO_MIN = 0.1
local UNSTABLE_REG_RDIST_MIN = 3.0

local smpMoving = 100


local n_smp = 25     -- # bg samples
local req_smp = 2    -- required # bg samples

local n_fm = 64


local COLOR_RANGE = 255

local kernel = torch.ByteTensor(5, 5):fill(1)


function BGSubtractor:__init(img, net_dir)

  -- // LOAD NETWORK //
  self.model = torch.load(net_dir)
  self.model = self.model.modules[1].modules[1]
  self.model.modules[#self.model] = nil
  self.model.modules[#self.model] = nil
  self.model:add(nn.Normalize2())

  -- print(self.model)

  for i = 1, #self.model.modules do
    local m = self.model:get(i)
    if torch.typename(m) == 'cudnn.SpatialConvolution' or torch.typename(m) == 'cudnn.SpatialMaxPooling' then
      m.padW = 1
      m.padH = 1
    end
  end

  self.model:cuda()

  -- // INITIALIZE //
  self.src  = torch.CudaTensor(1, 1, img:size(1), img:size(2))

  self.BGModel_c    = torch.CudaTensor(n_smp, 1, img:size(1), img:size(2))-- Background model
  self.BGModel_desc = torch.CudaTensor(n_smp, 1, n_fm, img:size(1), img:size(2))
  self.vols         = torch.CudaTensor(1, 1, img:size(1), img:size(2))
  self.cvols        = torch.CudaTensor(1, 1, img:size(1), img:size(2))
  self.cDist        = torch.CudaTensor(img:size(1), img:size(2))
  self.dDist        = torch.CudaTensor(1, 1, img:size(1), img:size(2))

  self.idx          = 0

  self.T            = torch.CudaTensor(img:size(1), img:size(2)):fill(FB_T_LOWER) -- T(x) learning rate
  self.R            = torch.CudaTensor(img:size(1), img:size(2)):fill(1) -- R(x)  threshold
  self.v            = torch.CudaTensor(img:size(1), img:size(2)):fill(10) -- v(x)
  self.d_m          = torch.CudaTensor(img:size(1), img:size(2)) -- D_last(x)  Dynamic ctrl
  self.D_m_LT       = torch.CudaTensor(img:size(1), img:size(2)):fill(0) -- D_min(x) LT
  self.D_m_ST       = torch.CudaTensor(img:size(1), img:size(2)):fill(0) -- D_min(x) ST
  self.rSgm_LT      = torch.CudaTensor(img:size(1), img:size(2)):fill(0) --s_t LT
  self.rSgm_ST      = torch.CudaTensor(img:size(1), img:size(2)):fill(0) --s_t ST
  self.Sgm_LT       = torch.CudaTensor(img:size(1), img:size(2)):fill(0)
  self.Sgm_ST       = torch.CudaTensor(img:size(1), img:size(2)):fill(0)
  self.unstable     = torch.CudaTensor(img:size(1), img:size(2)):fill(0)
  self.Blink        = torch.CudaTensor(img:size(1), img:size(2)):fill(0)
  self.curBlink     = torch.CudaTensor(img:size(1), img:size(2)):fill(0)
  self.lastBlink    = torch.CudaTensor(img:size(1), img:size(2)):fill(0)

  -- temp tensor
  self.cur_R_c      = torch.CudaTensor(img:size(1), img:size(2))
  self.cur_R_d      = torch.CudaTensor(img:size(1), img:size(2))

  self.sumDist      = torch.CudaTensor(img:size(1), img:size(2))
  self.n_goodMap    = torch.CudaTensor(n_smp, 1, img:size(1), img:size(2))
  self.cnt          = torch.CudaTensor(img:size(1), img:size(2))

  self.curFGMask    = torch.CudaTensor(img:size(1), img:size(2)):fill(0)
  self.lastFG       = torch.CudaTensor(img:size(1), img:size(2)):fill(0)
  self.lastRawFG       = torch.CudaTensor(img:size(1), img:size(2)):fill(0)
  self.FG_inv       = torch.CudaTensor(img:size(1), img:size(2)):fill(0)
  self.lastFG_inv   = torch.CudaTensor(img:size(1), img:size(2)):fill(0)

  self.tmp    = torch.CudaTensor(img:size(1), img:size(2))

  -- random mask
  self.r_smp = torch.CudaTensor(img:size(1), img:size(2))    -- random an index of sample in the model to be replaced
  self.r_prb = torch.CudaTensor(img:size(1), img:size(2))    -- random an probability to do the replacement

  self.src:copy(img)


  local output = self.model:forward(self.src)

  for i = 1, n_smp do
    self.BGModel_desc[i]:copy(output[1])
    self.BGModel_c[i]:copy(self.src[{ {}, {}, {} }])
  end

  collectgarbage()
end

function BGSubtractor:operator(input, trigger)
  local tk = sys.clock()
  self.src:copy(input)

  -- 1. set time subsampling factor
  self.idx = self.idx + 1
  local alpha_LT = 1/math.min(self.idx, smpMoving)
  local alpha_ST = 1/math.min(self.idx, smpMoving/4)
  self.curFGMask:fill(0)

  -- 2. Compute the current thresholds R_c and R_d
  cutils.update_threshold(self.R, self.cur_R_c, self.cur_R_d, self.unstable)

  -- 3. check unstable regions: unstable = 1, stable = 0
  cutils.check_unstable(self.unstable, self.R, self.rSgm_LT, self.rSgm_ST, self.Sgm_LT, self.Sgm_ST)

  -- 4. extract features from input image
  local output = self.model:forward(self.src)

  --  count good samples
  local mask = self.vols[1]

  self.d_m:fill(3)
  self.tmp:fill(0)
  self.n_goodMap:fill(0)

  for i = 1, n_smp do
    -- compute the similarity btw current image and each BG sample
    self.cvols:csub(self.src, self.BGModel_c[i]):abs() -- difference in color
    self.cnt:lt(self.cvols, self.cur_R_c)

    cutils.computeDescDist(output[{{1}}], self.BGModel_desc[i], self.vols[{{1}}]) -- difference emb
    self.n_goodMap[i]:lt(mask, self.cur_R_d):add(self.cnt):ge(1)
    self.cvols:div(COLOR_RANGE):mul(0.5)
    mask:mul(0.5)
    self.cvols:add(mask)
    self.d_m:cmin(self.cvols)  -- take aveage with desc difference
  end


  -- 4. count the number of good samples
  self.cnt:sum(self.n_goodMap, 1)

  -- 6. // BG/FG CLASSIFICATION //
  self.curFGMask:lt(self.cnt, req_smp)


  -- update BG model, replace a random sample in the model with a current sample
  self.r_smp:uniform(0, n_smp-1)
  self.r_prb:uniform(1, 32767)
  cutils.update_model(self.BGModel_desc, self.BGModel_c, output,
                      self.src, self.T, self.curFGMask, self.r_smp, self.r_prb,
                      trigger and 1 or 0)

  cutils.update_params(
    alpha_LT,
    alpha_ST,
    self.R,
    self.T,
    self.v,
    self.D_m_LT,
    self.D_m_ST,
    self.rSgm_LT,
    self.rSgm_ST,
    self.Sgm_LT,
    self.Sgm_ST,
    self.unstable,
    self.curFGMask,
    self.lastFG,
    self.Blink,
    self.d_m,
    self.cnt)

  -- 12. // POST-PROCESSING //
  -- Step 1. Estimate blinking pixels
  util.bitwise_xor(self.curFGMask, self.lastRawFG, self.curBlink)
  util.bitwise_or(self.curBlink, self.lastBlink, self.Blink)
  self.lastBlink:copy(self.curBlink)
  self.lastRawFG:copy(self.curFGMask)

  -- Step 2. Morphology operation
  cutils.median2d(self.curFGMask, self.curFGMask, 9)
  cutils.binary_dilate(self.curFGMask, self.FG_inv, 3)
  util.bitwise_and(self.Blink, self.lastFG_inv, self.Blink)
  util.bitwise_not(self.FG_inv, self.lastFG_inv)
  util.bitwise_and(self.Blink, self.lastFG_inv, self.Blink)

  cutils.binary_erode(self.curFGMask, self.curFGMask, 3)
  cutils.binary_dilate(self.curFGMask, self.curFGMask, 3)

  --update mean seg
  cutils.update_seg(self.Sgm_LT, self.Sgm_ST, self.curFGMask, alpha_LT, alpha_ST)

  -- // LAST = CURRENT for the next iter
  self.lastFG:copy(self.curFGMask)

  -- collectgarbage()
  print(sys.clock() - tk)

end

function BGSubtractor:getFG()
  return self.curFGMask:view(1, self.src:size(3), self.src:size(4)):float()
end

return M.BGSubtractor
