for i in {1..20}
do
    f=$(ls ../Chapter$i | grep tex | sed 's/.tex//')
    echo $f
    python3 parse_gls.py ../Chapter$i/$f.tex | sed 's/^\(%.*\)/<!-- \1 -->/' | sed '/!Mode/d;/Translator/d;s/\\chapter{\(.*\)}/---\ntitle: \1\nlayout: post\nshare: false\n---/' | sed 's/^\\section{\(.*\)}/\n# \1\n/' |  sed 's/^\\subsection{\(.*\)}/\n## \1\n/' | sed '/\\label/d' | sed 's/<BAD>//g;s/<bad>//g;s/``/"/g;s/'\'''\''/"/g' | sed 's/\\citep{[^}]*}/{cite?}/g;s/\\citet//g;s/\\cite//g' | sed 's/ref{[^}]*}/?/g;s/\\ENNAME{\([^}]*\)}/\1/g;s/\\NUMTEXT{\([^}]*\)}/\1/g;' | sed 's/\\begin{itemize}//;s/\\end{itemize}//' | sed 's/.*\\item/\+/' > ../docs/_posts/2016-12-$(printf "%02d" $i)-Chapter${i}_$f.md
done
