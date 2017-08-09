local util = {}

require 'torch'
require 'cutorch'

function util.bitwise_and(src1, src2, des)
  des:cmul(src1, src2):eq(1)
  -- return des
end

function util.bitwise_or(src1, src2, des)
  des:add(src1, src2):ge(1)
  -- return des
end

function util.bitwise_xor(src1, src2, des)
  des:add(src1, src2):eq(1)
  -- return des
end

function util.bitwise_not(src, des)
  des:eq(src, 0)
  -- return des
end

return util
