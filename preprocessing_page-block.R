library(caret)
library(psych)  
library(plyr)
library(qdap)
library(stringr)

con <- file("page-blocks.data", "r")
cols <- c("height","length","area","eccen","p_black","p_and","mean_tr","blackpix","blackand","wb_trans","class")
instances <- length(readLines("page-blocks.data"))
df <- data.frame(matrix(ncol = length(cols), nrow = instances))
colnames(df) <- cols
nr<-1
while ( nr<=instances ) {
  line = readLines(con, n = 1)
  if ( length(line) == 0 ) {
    break
  }
  line <- str_trim(clean(line))
  x <- unlist((strsplit(line," ")))
  #x<-lapply(x, function(y) y[!is.na(y)])
  for(i in 1:length(cols)) {
    df[nr, cols[i]] <- as.numeric(x[i])
  }
  nr=nr+1
}
close(con)
save(df, file = "page-blocks.RData")
write.csv(df, 'page-blocks.csv')