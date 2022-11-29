from textblob import TextBlob
import sys



if __name__ == "__main__":
    text = " ".join(sys.argv[1:])

    print("Texte : {}".format(text))
    print("="*30)
    print( "Subjectivity : {} ; Polarity : {}".format( TextBlob(text).sentiment.subjectivity , TextBlob(text).sentiment.polarity) )
