# lob-learn
Reinforcement learning for Limit Order Books trade execution

### The Project
We attempt to implement the findings of [this](https://www.cis.upenn.edu/~mkearns/papers/rlexec.pdf) paper.

### Data
The data we are using are located in `data/order_books/` (please ignore the other data). These are Chinese stock market futures, where the file "IFXXYY" is the order book data for the futures contract maturing in the YY month of the XX year. Such a file name could exist in two different folders. For example, the 2013 June contract traded in both 2012 and 2013, so in both the 2012 and 2013 folders there is a file called "IF1306."

Note that the individual data files are actually zip-compressed. In order to actually view the contents, uncompress the files by renaming them with a ".zip" extension and then extracting them, or by dragging them all into a tool like [the Unarchiver](https://theunarchiver.com/) to quickly extract all of them at once.