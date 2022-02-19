lat2asc = {
            'Dž': 'Dy',
            'Lj': 'Lx',
            'Nj': 'Nx',
            'DŽ': 'DY',
            'LJ': 'LX',
            'NJ': 'NX',
            'lj': 'lx',
            'nj': 'nx',
            'dž': 'dy',
            'Č': 'Cy',
            'Ć': 'Cx',
            'Ž': 'Zx',
            'Đ': 'Dx',
            'Š': 'Sx',
            'č': 'cy',
            'ć': 'cx',
            'ž': 'zx',
            'đ': 'dx',
            'š': 'sx'
        }
lat2cyr = {
            'LJ': 'Љ',
            'NJ': 'Њ',
            'DŽ': 'Џ',
            'Lj': 'Љ',
            'Nj': 'Њ',
            'Dž': 'Џ',
            'A': 'А',
            'B': 'Б',
            'V': 'В',
            'G': 'Г',
            'D': 'Д',
            'Đ': 'Ђ',
            'E': 'Е',
            'Ž': 'Ж',
            'Z': 'З',
            'I': 'И',
            'J': 'Ј',
            'K': 'К',
            'L': 'Л',
            'M': 'М',
            'N': 'Н',
            'O': 'О',
            'P': 'П',
            'R': 'Р',
            'S': 'С',
            'T': 'Т',
            'Ć': 'Ћ',
            'U': 'У',
            'F': 'Ф',
            'H': 'Х',
            'C': 'Ц',
            'Č': 'Ч',
            'Š': 'Ш',
            'lj': 'љ',
            'nj': 'њ',
            'dž': 'џ',
            'a': 'а',
            'b': 'б',
            'v': 'в',
            'g': 'г',
            'd': 'д',
            'đ': 'ђ',
            'e': 'е',
            'ž': 'ж',
            'z': 'з',
            'i': 'и',
            'j': 'ј',
            'k': 'к',
            'l': 'л',
            'm': 'м',
            'n': 'н',
            'o': 'о',
            'p': 'п',
            'r': 'р',
            's': 'с',
            't': 'т',
            'ć': 'ћ',
            'u': 'у',
            'f': 'ф',
            'h': 'х',
            'c': 'ц',
            'č': 'ч',
            'š': 'ш'
        }
cyr2asc = {
            'Љ': 'Lx',
            'Њ': 'Nx',
            'Џ': 'Dy',
            'А': 'A',
            'Б': 'B',
            'В': 'V',
            'Г': 'G',
            'Д': 'D',
            'Ђ': 'Dx',
            'Е': 'E',
            'Ж': 'Zx',
            'З': 'Z',
            'И': 'I',
            'Ј': 'J',
            'К': 'K',
            'Л': 'L',
            'М': 'M',
            'Н': 'N',
            'О': 'O',
            'П': 'P',
            'Р': 'R',
            'С': 'S',
            'Т': 'T',
            'Ћ': 'Cx',
            'У': 'U',
            'Ф': 'F',
            'Х': 'H',
            'Ц': 'C',
            'Ч': 'Cy',
            'Ш': 'Sx',
            'љ': 'lx',
            'њ': 'nx',
            'џ': 'dy',
            'а': 'a',
            'б': 'b',
            'в': 'v',
            'г': 'g',
            'д': 'd',
            'ђ': 'dx',
            'е': 'e',
            'ж': 'zx',
            'з': 'z',
            'и': 'i',
            'ј': 'j',
            'к': 'k',
            'л': 'l',
            'м': 'm',
            'н': 'n',
            'о': 'o',
            'п': 'p',
            'р': 'r',
            'с': 's',
            'т': 't',
            'ћ': 'cx',
            'у': 'u',
            'ф': 'f',
            'х': 'h',
            'ц': 'c',
            'ч': 'cy',
            'ш': 'sx'
        }
asc2lat = {v: k for k, v in lat2asc.items()}
cyr2lat = {v: k for k, v in lat2cyr.items()}
asc2cyr = {v: k for k, v in cyr2asc.items()}


def convert(text, ctype, direction=""):
    dic = {}
    if ctype == 'LATtoASC':
        dic = lat2asc
    if ctype == 'ASCtoLAT':
        dic = asc2lat
    if ctype == 'LATtoCYR':
        dic = lat2cyr
    if ctype == 'CYRtoLAT':
        dic = cyr2lat
    if ctype == 'CYRtoASC':
        dic = cyr2asc
    if ctype == 'ASCtoCYR':
        dic = asc2cyr

    if direction == 'back':
        for key in dic.keys():
            text = text.replace(dic[key], key)
    else:
        for key in dic.keys():
            text = text.replace(key, dic[key])

    return text
