#!/usr/bin/env python3
import sys
import re

def create_gls_dict():
    term_file = open('../terminology.tex', 'r')
    term_file.readline()
    terms = term_file.read()
    term_file.close()
    term_name_dict = {}
    term_symbol_dict = {}
    for term in terms.split('\n\n'):
        tmp = term.strip().split('\n')
        if tmp[2].strip()[:4] != 'name':
            print(tmp)
        dirty_term = tmp[0][18:]
        term_name_dict[dirty_term[:dirty_term.find('}')]] = tmp[2].strip()[5:-1]
        for s in tmp:
            if s.strip()[:6] == 'symbol':
                dirty_symbol = s.strip()[8:]
                term_symbol_dict[dirty_term[:dirty_term.find('}')]] =  dirty_symbol[:dirty_symbol.find('}')]
                

    return term_name_dict, term_symbol_dict


def replace_single_gls(tex, p, term_dict):
    pl = p.finditer(tex)
    new_tex = ''
    prev_pos = 0
    for i in pl:
        new_tex += tex[prev_pos:i.start()] + term_dict[i.group()[i.group().find('{')+1:-1]]
        prev_pos = i.end()
    new_tex += tex[prev_pos:]
    return new_tex


def replace_all_gls(input_tex, term_name_dict, term_symbol_dict):
    # matched \gls{}
    tex_file = open(input_tex, 'r')
    tex = tex_file.read()

    p = re.compile('\\\\gls\\{[^\\}]*\\}')
    tex = replace_single_gls(tex, p, term_name_dict)

    p = re.compile('\\\\firstgls\\{[^\\}]*\\}')
    tex = replace_single_gls(tex, p, term_name_dict)

    p = re.compile('\\\\firstall\\{[^\\}]*\\}')
    tex = replace_single_gls(tex, p, term_name_dict)

    p = re.compile('\\\\firstacr\\{[^\\}]*\\}')
    tex = replace_single_gls(tex, p, term_name_dict)

    p = re.compile('\\\\glsacr\\{[^\\}]*\\}')
    tex = replace_single_gls(tex, p, term_name_dict)

    p = re.compile('\\\\glsentrytext\\{[^\\}]*\\}')
    tex = replace_single_gls(tex, p, term_name_dict)

    p = re.compile('\\\\glssymbol\\{[^\\}]*\\}')
    tex = replace_single_gls(tex, p, term_symbol_dict)

    return tex

if __name__ == '__main__':
    input_tex = sys.argv[1]
    term_name_dict, term_symbol_dict = create_gls_dict()
    print(replace_all_gls(input_tex, term_name_dict, term_symbol_dict))
    #print(term_name_dict, term_symbol_dict)

