set.seed(20000)
data=data.frame(A=rpois(900,3),B=rnorm(900),C=runif(900))

jpeg(file="rplot.jpeg",width=600,height=350)
#attach(mtcars)
par(mfrow=c(1,2))

boxplot(data);boxplot(data,yaxt="n")
dev.off()
#temp