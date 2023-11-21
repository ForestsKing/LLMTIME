import numpy as np


class Scaler:
    def __init__(self, history, alpha=0.95, beta=0.3):
        self.min_ = np.min(history) - beta * (np.max(history) - np.min(history))
        self.q = np.quantile(history - self.min_, alpha)

    def transform(self, x):
        return (x - self.min_) / self.q

    def inv_transform(self, x):
        return x * self.q + self.min_


class Serializer:
    def __init__(
            self, base=10, prec=3, max_val=10000000.0,
            time_sep=' ,', bit_sep=' ', plus_sign='', minus_sign=' -', decimal_point='', missing_str=' Nan'):

        self.max_val = max_val
        self.base = base
        self.prec = prec
        self.bit_sep = bit_sep
        self.decimal_point = decimal_point
        self.missing_str = missing_str
        self.minus_sign = minus_sign
        self.plus_sign = plus_sign
        self.time_sep = time_sep

    def tokenize(self, arr):
        return ''.join([self.bit_sep + str(b) for b in arr])

    def num2repr(self, val):
        sign = 1 * (val >= 0) - 1 * (val < 0)
        val = np.abs(val)

        max_bit_pos = int(np.ceil(np.log(self.max_val) / np.log(self.base)).item())
        before_decimals = []
        for i in range(max_bit_pos):
            digit = (val / self.base ** (max_bit_pos - i - 1)).astype(int)
            before_decimals.append(digit)
            val -= digit * self.base ** (max_bit_pos - i - 1)
        before_decimals = np.stack(before_decimals, axis=-1)

        after_decimals = []
        for i in range(self.prec):
            digit = (val / self.base ** (-i - 1)).astype(int)
            after_decimals.append(digit)
            val -= digit * self.base ** (-i - 1)
        after_decimals = np.stack(after_decimals, axis=-1)
        digits = np.concatenate([before_decimals, after_decimals], axis=-1)

        return sign, digits

    def repr2num(self, sign, digits):
        D = digits.shape[-1]
        digits_flipped = np.flip(digits, axis=-1)
        powers = -np.arange(-self.prec, -self.prec + D)
        val = np.sum(digits_flipped / self.base ** powers, axis=-1)
        val += 0.5 / self.base ** self.prec

        return sign * val

    def serialize(self, arr):
        ismissing = np.isnan(arr)
        sign_arr, digits_arr = self.num2repr(np.where(np.isnan(arr), np.zeros_like(arr), arr))

        bit_strs = []
        for sign, digits, missing in zip(sign_arr, digits_arr, ismissing):
            nonzero_indices = np.where(digits != 0)[0]
            if len(nonzero_indices) == 0:
                digits = np.array([0])
            else:
                digits = digits[nonzero_indices[0]:]

            if len(self.decimal_point):
                digits = np.concatenate([digits[:-self.prec], np.array([self.decimal_point]), digits[-self.prec:]])

            digits = self.tokenize(digits)
            sign_sep = self.plus_sign if sign == 1 else self.minus_sign

            if missing:
                bit_strs.append(self.missing_str)
            else:
                bit_strs.append(sign_sep + digits)

        bit_str = self.time_sep.join(bit_strs)
        bit_str += self.time_sep

        return bit_str

    def deserialize(self, bit_str):
        bit_strs = bit_str.split(self.time_sep)
        bit_strs = [a for a in bit_strs if len(a) > 0]

        sign_arr, digits_arr = [], []
        for i, bit_str in enumerate(bit_strs):
            if bit_str.startswith(self.minus_sign):
                sign = -1
            else:
                sign = 1

            bit_str = bit_str[len(self.plus_sign):] if sign == 1 else bit_str[len(self.minus_sign):]
            if self.bit_sep == '':
                bits = [b for b in bit_str.lstrip()]
            else:
                bits = [b[:1] for b in bit_str.lstrip().split(self.bit_sep)]

            digits = []
            for b in bits:
                if b == self.decimal_point:
                    continue

                if b.isdigit():
                    digits.append(int(b))
                else:
                    break

            sign_arr.append(sign)
            digits_arr.append(digits)

        max_len = max([len(d) for d in digits_arr])
        for i in range(len(digits_arr)):
            digits_arr[i] = [0] * (max_len - len(digits_arr[i])) + digits_arr[i]

        return self.repr2num(np.array(sign_arr), np.array(digits_arr))
