#!/usr/bin/perl -w

use strict;

if (!scalar(@ARGV)) {
  print STDERR "Syntax: multi-bleu.perl [-length_analysis bucket] [ref-stem] < [system-output]
If one reference translation: ref-stem is filename
If multiple reference translations: ref-stem[0,1,2,...] is filename\n";
}

my $length_analysis;
if ($ARGV[0] eq '-length_analysis') {
  shift @ARGV;
  $length_analysis = shift @ARGV;
}

my @CORRECT_BUCKET;
my @TOTAL_BUCKET;
my @COUNT_LENGTH;
my $max_bucket=0;

my $stem = $ARGV[0];
my @REF;
my $ref=0;
while(-e "$stem$ref") {
    &add_to_ref("$stem$ref",\@REF);
    $ref++;
}
&add_to_ref($stem,\@REF) if -e $stem;
die("did not find any reference translations at $stem") unless scalar @REF;

sub add_to_ref {
    my ($file,$REF) = @_;
    my $s=0;
    open(REF,$file);
    while(<REF>) {
    chop;
    push @{$$REF[$s++]}, $_;
    }
    close(REF);
}

my(@CORRECT,@TOTAL,$length_translation,$length_reference);
my $s=0;
while(<STDIN>) {
    chop;
    my @WORD = split;
    my %REF_NGRAM = ();
    my $length_translation_this_sentence = scalar(@WORD);
    my ($closest_diff,$closest_length) = (9999,9999);
    my $bucket;
    foreach my $reference (@{$REF[$s]}) {
#      print "$s $_ <=> $reference\n";
    my @WORD = split(/ /,$reference);
    my $length = scalar(@WORD);
    if ($length_analysis) {
        $bucket = int($length/$length_analysis);
        $max_bucket=$bucket if ($bucket>$max_bucket);
    }
    if (abs($length_translation_this_sentence-$length) < $closest_diff) {
        $closest_diff = abs($length_translation_this_sentence-$length);
        $closest_length = $length;
#    print "$i: closest diff = abs($length_translation_this_sentence-$length)<BR>\n";
    }
    for(my $n=1;$n<=4;$n++) {
        my %REF_NGRAM_N = ();
        for(my $start=0;$start<=$#WORD-($n-1);$start++) {
        my $ngram = "$n";
        for(my $w=0;$w<$n;$w++) {
            $ngram .= " ".$WORD[$start+$w];
        }
        $REF_NGRAM_N{$ngram}++;
        }
        foreach my $ngram (keys %REF_NGRAM_N) {
        if (!defined($REF_NGRAM{$ngram}) ||
            $REF_NGRAM{$ngram} < $REF_NGRAM_N{$ngram}) {
            $REF_NGRAM{$ngram} = $REF_NGRAM_N{$ngram};
#        print "$i: REF_NGRAM{$ngram} = $REF_NGRAM{$ngram}<BR>\n";
        }
        }
    }
    }
    if ($bucket) {
        $COUNT_LENGTH[$bucket]++;
    }
    $length_translation += $length_translation_this_sentence;
    $length_reference += $closest_length;
    for(my $n=1;$n<=4;$n++) {
    my %T_NGRAM = ();
    for(my $start=0;$start<=$#WORD-($n-1);$start++) {
        my $ngram = "$n";
        for(my $w=0;$w<$n;$w++) {
        $ngram .= " ".$WORD[$start+$w];
        }
        $T_NGRAM{$ngram}++;
    }
    foreach my $ngram (keys %T_NGRAM) {
        $ngram =~ /^(\d+) /;
        my $n = $1;
#    print "$i e $ngram $T_NGRAM{$ngram}<BR>\n";
        $TOTAL[$n] += $T_NGRAM{$ngram};
        if ($bucket) {
            $TOTAL_BUCKET[$bucket][$n] += $T_NGRAM{$ngram};
        }
        if (defined($REF_NGRAM{$ngram})) {
        if ($REF_NGRAM{$ngram} >= $T_NGRAM{$ngram}) {
            if ($bucket) {
                $CORRECT_BUCKET[$bucket][$n] += $T_NGRAM{$ngram};
            }
            $CORRECT[$n] += $T_NGRAM{$ngram};
#        print "$i e correct1 $T_NGRAM{$ngram}<BR>\n";
        }
        else {
            if ($bucket) {
                $CORRECT_BUCKET[$bucket][$n] += $REF_NGRAM{$ngram};
            }
            $CORRECT[$n] += $REF_NGRAM{$ngram};
#        print "$i e correct2 $REF_NGRAM{$ngram}<BR>\n";
        }
        }
    }
    }
    $s++;
}
my $brevity_penalty = 1;
if ($length_translation<$length_reference) {
    $brevity_penalty = exp(1-$length_reference/$length_translation);
}
my $bleu = $brevity_penalty * exp((my_log( $CORRECT[1]/$TOTAL[1] ) +
                   my_log( $CORRECT[2]/$TOTAL[2] ) +
                   my_log( $CORRECT[3]/$TOTAL[3] ) +
                   my_log( $CORRECT[4]/$TOTAL[4] ) ) / 4);

printf "BLEU = %.2f, %.1f/%.1f/%.1f/%.1f (BP=%.3f, ration=%.3f)\n",
    100*$bleu,
    100*$CORRECT[1]/$TOTAL[1],
    100*$CORRECT[2]/$TOTAL[2],
    100*$CORRECT[3]/$TOTAL[3],
    100*$CORRECT[4]/$TOTAL[4],
    $brevity_penalty,
    $length_translation / $length_reference;

if ($length_analysis) {
    print "\nLENGTH ANALYSIS:\n";
    for(my $b=int(1/$length_analysis); $b<=$max_bucket; $b++) {
        my $range=$b;
        if ($length_analysis != 1) {
            $range=($b*$length_analysis+1)."-".(($b+1)*$length_analysis);
        }
        print "$range";;
        if ($TOTAL_BUCKET[$b] && $TOTAL_BUCKET[$b][4] && $CORRECT_BUCKET[$b][4]) {
          printf "\t%d\t%.2f", $COUNT_LENGTH[$b],
                   100*$brevity_penalty * exp((my_log( $CORRECT_BUCKET[$b][1]/$TOTAL_BUCKET[$b][1] ) +
                   my_log( $CORRECT_BUCKET[$b][2]/$TOTAL_BUCKET[$b][2] ) +
                   my_log( $CORRECT_BUCKET[$b][3]/$TOTAL_BUCKET[$b][3] ) +
                   my_log( $CORRECT_BUCKET[$b][4]/$TOTAL_BUCKET[$b][4] ) ) / 4);
        }
        print "\n";
    }
}

sub my_log {
  return -9999999999 unless $_[0];
  return log($_[0]);
}
