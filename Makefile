all:
	xelatex dlbook_cn.tex && bibtex dlbook_cn.aux && texindy dlbook_cn.idx && makeglossaries dlbook_cn && xelatex dlbook_cn.tex && xelatex dlbook_cn.tex

clean:
	find . -type f -iregex '.*\.\(aux\|log\|toc\|backup\|acr\|brf\|gz\|acn\|xdy\|alg\)$$'  -delete
