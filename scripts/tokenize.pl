#!/usr/bin/perl

########################################################################
#                                                                      #
#  tokenization script for tagger preprocessing                        #
#  Author: Helmut Schmid, IMS, University of Stuttgart                 #
#          Serge Sharoff, University of Leeds                          #
#  Description:                                                        #
#  - splits input text into tokens (one token per line)                #
#  - cuts off punctuation, parentheses etc.                            #
#  - disambiguates periods                                             #
#  - preserves SGML markup                                             #
#                                                                      #
########################################################################

use Getopt::Std;
use utf8;
use Encode;

getopts('chgfeipzIFa:r');

# Modify the following lines in order to adapt the tokenizer to other
# types of text and/or languages

# characters which have to be cut off at the beginning of a word
my $PChar='[¿¡{\'\\`"‚„†‡‹‘’“”•–—›»«';

# characters which have to be cut off at the end of a word
my $FChar=']}\'\`\",;:\!\?\%‚„…†‡‰‹‘’“”•–—›»«';

# character sequences which have to be cut off at the beginning of a word
my $PClitic='';

# character sequences which have to be cut off at the end of a word
my $FClitic='';


if (defined($opt_e)) {
    # English
    $FClitic = "['’´](?:s|re|ve|d|m|em|ll)|n['’´]t";
}

### NO MODIFICATIONS REQUIRED BEYOND THIS LINE #########################

if (defined($opt_h)) {
    die "
Usage: utf8-tokenize.perl [ options ] ...files...

Options:
-e : English text 
-i : Italian text
-f : French text
-z : Galician text
-a <file>: <file> contains a list of words which are either abbreviations or
           words which should not be further split.
";
}

# Read the list of abbreviations and words
if (defined($opt_a)) {
    die "Can't read: $opt_a: $!\n"  unless (open(FILE, $opt_a));
    while (<FILE>) {
	$_ = decode('utf8',$_);
	s/^[ \t\r\n]+//;
	s/[ \t\r\n]+$//;
	next if (/^\#/ || /^\s$/);    # ignore comments
	$Token{$_} = 1;
    }
    close FILE;
}

#SS: main loop; 
my $first_line = 1;
while (<>) {

    my $output = "";
    $_ = decode('utf8',$_);
    if (/^<.*>$/){
    print encode('utf-8', $_);
    }
    else{
    # delete optional byte order markers (BOM)
    if ($first_line) {
	undef $first_line;
	s/^\x{FEFF}//;
    }

    @S = split();
    for( $i=0; $i<=$#S; $i++) {
	$_ = $S[$i];
	
	# add a blank at the beginning and the end of each segment
	$_ = " $_ ";
	# insert missing blanks after punctuation
	s/(\.\.\.)/ ... /g;
	s/([;\!\?])([^ ])/$1 $2/g;
	    
	@F = split;
	for ( $j=0; $j<=$#F; $j++) {
	    my $suffix = "";

	    $_ = $F[$j];

	    # add newline after sentence-final punctuation
	    $suffix = "\n" if /^(.*[^.])?[.!?][)\]\'\"]*$/ or /।$/; # The latter is Hindi
	    # separate punctuation and parentheses from words
	    $finished = 0;
	    while (!$finished) {

		# preceding parentheses
		if (s/^(\()([^\)]*)(.)$/$2$3/) {
		    $output .= "$1\n";
		}
		
		# following preceding parentheses
		elsif (s/^([^(]+)(\))$/$1/) {
		    $suffix = "$2\n$suffix";
		}
		
		# preceding double quotation
		elsif (s/^(``|,,)(.)/$2/) {
		    $output .= "$1\n";
		}
		
		# following double quotation
		elsif (s/(.)('')$/$1/) {
		    $suffix = "$2\n$suffix";
		}
		
		elsif (s/^([$PChar])(.)/$2/) {
		    $output .= "$1\n";
		}
		
		# cut off preceding punctuation
		elsif (s/^([$PChar])(.)/$2/) {
		    $output .= "$1\n";
		}
		
		# cut off trailing punctuation
		elsif (s/(.)([$FChar])$/$1/) {
		    $suffix = "$2\n$suffix";
		}
		
		# cut off trailing periods if punctuation precedes
		elsif (s/([$FChar]|\))\.$//) { 
		    $suffix = ".\n$suffix";
		    if ($_ eq "") {
			$_ = $1;
		    } else {
			$suffix = "$1\n$suffix";
		    }
		}
		
		else {
		    $finished = 1;
		}
	    }
	    
	    # handle explicitly listed tokens
	    if (defined($Token{$_})) {
		$suffix =~ s/^([)\]\'\"]*)\n$/$1/;  # remove previously added newline
		$output .= "$_\n$suffix";
		next;
	    }
	    
	    # abbreviations of the form A. or U.S.A.
	    if (/^(\p{L}\.)+$/ && $1 ne 'á') {
		$suffix =~ s/^([)\]\'\"]*)\n$/$1/;  # remove previously added newline
		$output .= "$_\n$suffix";
		next;
	    }
	    
	    
	    # disambiguate periods
	    if (/^(.+)\.$/ && $_ ne "...") {
		if ($opt_g && /^[0-9]+\.$/) {
		    $suffix =~ s/^([)\]\'\"]*)\n$/$1/;  # remove previously added newline
		}
		else {
		    $_ = $1;
		    $suffix = ".\n$suffix";
		    if (defined($Token{$_})) {
			$output .= "$_\n$suffix";
			next;
		    }
		}
	    }
	    
	    # cut off clitics
	    while (s/^(--)(.)/$2/) {
		$output .= "$1\n";
	    }
	    if ($PClitic ne '') {
		while (s/^($PClitic)(.)/$2/i) {
		    my $clitic = $1;
		    $clitic =~ tr/'’´/'''/;
		    $output .= "$clitic\n";
		}
	    }
	    
	    while (s/(.)(--)$/$1/) {
		$suffix = "$2\n$suffix";
	    }
	    if ($FClitic ne '') {
		while (s/(.)($FClitic)$/$1/i) {
		    my $clitic = $2;
		    $clitic =~ tr/'’´/'''/;
		    $suffix = "$clitic\n$suffix";
		}
	    }
	    
	    $output .= "$_\n$suffix";
	}
    }
    print encode('utf-8', $output);
}}

print "\n"
    
