/*
 * lodestone (let's hash books) 
 * @author: marin.silic@gmail.com
 */

import java.util.*;
import java.io.*;
import java.text.*;

import org.apache.commons.codec.binary.Hex;
import org.apache.commons.codec.digest.*;
import org.apache.commons.codec.*;

public class Lodestone {
  
  private static final String OUT_HASHES = "hashes_75k.txt";
  private static final int HASH_LENGTH = 128;
  private static final int BAND_CNT = 16;
  private static final int BAND_LENGTH = 8;
  private static final String SP = " ";
  private static final String IN_PATH = "75k/";
  private static final String NL = System.getProperty("line.separator");
  
  public static void main(String[] args) throws IOException, DecoderException {
    
    
    boolean cHash = Boolean.parseBoolean(args[0]);
    
    PrintWriter pwb = new PrintWriter(System.out);
    StringBuilder sbbrute = new StringBuilder();
    File folder = new File(IN_PATH);
    File[] files = folder.listFiles();
    int N = files.length;
    byte[][] hashes = new byte[N][BAND_CNT];
    
    String[] names = new String[N];
    Map<String, Integer> map = new HashMap<String, Integer>();
    
    Set<String> setStopWords = new HashSet<String>();
    for (String word : stopWords) {
      setStopWords.add(word);
    }
    int i;
    if (cHash) {
      i = 0;
      PrintWriter pwh = new PrintWriter(OUT_HASHES);
      int fcnt = files.length;
      for (File f : files) {
        
        BufferedReader br = new BufferedReader(new FileReader(f.getPath()));
        StringBuilder sb = new StringBuilder();
        String line;
        while ((line = br.readLine()) != null) {
          sb.append(line);
          sb.append(NL);
        }
        String text = sb.toString();
        String name = f.getName();
        String start = name.substring(0, name.indexOf("_"));
        names[i] = start;
        int cnt = 1;
        if (map.containsKey(start)) {
          cnt += map.get(start);
        }
        map.put(start, cnt);
        if (cHash) {
          hashes[i] = computeHash(text);
          String trimName = name.substring(0, name.indexOf("."));
          pwh.println(trimName + SP + Hex.encodeHexString(hashes[i]));
        }
        System.out.println("File : " + (i + 1) + "/" + fcnt +  "\t" + f.getPath());
        i++;
      }
      pwh.close();
    } else {
      BufferedReader br = new BufferedReader(new FileReader(OUT_HASHES));
      for (i = 0; i < N; i++) {
        String[] nh = br.readLine().split(SP);
        hashes[i] = Hex.decodeHex(nh[1].toCharArray());
        String name = nh[0];
        String start = name.substring(0, name.indexOf("_"));
        names[i] = start;
        int cnt = 1;
        if (map.containsKey(start)) {
          cnt += map.get(start);
        }
        map.put(start, cnt);
      }
    }
    Map<Integer, Set<Integer>> cands = 
      processBuckets(hashes);
    int Q = N;
    Query[] qs = new Query[Q];
    sbbrute.append("Bucketing:" + NL);
    for (int k = 0; k < 40; k++) {
      System.out.println("Processing k = " + k);
      for (i = 0; i < Q; i++) {
        int tid = i;
        int dist = k;
        Query q = new Query(i, tid, tid, dist);
        qs[i] = q;
      }
      sbbrute.append(k);
      long es = System.nanoTime();
      long es1 = System.nanoTime();
      sbbrute.append(processQueries(cands, qs, hashes, names, map));
      long e = System.nanoTime();
      sbbrute.append("\t" + ((e - es)/(1000 * 1000)));
      sbbrute.append(NL);
    }
    sbbrute.append(NL);
    sbbrute.append("Brute Force:" + NL);
    for (int k = 0; k < 40; k++) {
      for (i = 0; i < Q; i++) {
        int tid = i;
        int dist = k;
        Query q = new Query(i, tid, tid, dist);
        qs[i] = q;
      }
      System.out.println("Processing Brute k = " + k);
      sbbrute.append(k);
      long bs = System.nanoTime();
      sbbrute.append(processQueriesBrute(qs, hashes, names, map));
      long be = System.nanoTime();
      sbbrute.append("\t" + ((be - bs)/(1000 * 1000)));
      sbbrute.append(NL);
    }
    
    pwb.println(sbbrute.toString());
    pwb.close();
  }
  
  private static String processQueriesBrute(Query[] qs, 
                                            byte[][] hashes,
                                            String[] names, 
                                            Map<String, Integer> map) {
    StringBuilder sb = new StringBuilder();
    int N = hashes.length;
    double pr = 0.0;
    double rc = 0.0;
    int cnt = 0;
    for (Query q : qs) {
      int tid = q.getTextId();
      int hammDist = q.getHammingDistance();
      byte[] thash = hashes[tid];
      List<Integer> tids = new ArrayList<Integer>();
      int truep = 0; 
      for (int i = 0; i < N; i++) {
        if (i != tid) {
          byte[] hash = hashes[i];
          if (areNearDuplicates(thash, hash, hammDist)) {
            tids.add(i);
            if (names[tid].equals(names[i])) {
              truep++;
            }
          }
        }
      }
      Collections.sort(tids);
      int T = tids.size();
      /*for (int i = 0; i < T; i++) {
        sb.append(tids.get(i));
        if (i != T - 1) {
          sb.append(SP);
        }
      }
      */
      //sb.append(T);
      //sb.append(NL);
      if (T != 0) {
        pr += (double) truep / (double) T;
        int all = map.get(names[tid]);
        rc += (double) truep / (double) (all - 1);
      } else {
        pr += 1.0;
        rc += 0.0;
      }
      cnt++;
    }
    DecimalFormat df = new DecimalFormat();
    df.setMaximumFractionDigits(2);
    pr /= (double) cnt;
    rc /= (double) cnt;
    sb.append("\t" + df.format(pr));
    sb.append("\t" + df.format(rc));
    sb.append("\t" + df.format((2 * (pr * rc) / (pr + rc))));
    return sb.toString();
  }
  
  private static Map<Integer, Set<Integer>> processBuckets( 
                                             byte[][] hashes) {
    int N = hashes.length;
    Map<Integer, Set<Integer>> buckets;
    Map<Integer, Set<Integer>> cands = new HashMap<Integer, Set<Integer>>();
    for (int i = 0; i < BAND_CNT; i += 4) {
      System.out.println("Processing bucket = " + i);
      buckets = new HashMap<Integer, Set<Integer>>();
      for (int j = 0; j < N; j++) {
        int b = (int) hashes[j][i];
        b <<= BAND_LENGTH;
        b += (int) hashes[j][i + 1];
        b <<= BAND_LENGTH;
        b += (int) hashes[j][i + 2];
        b <<= BAND_LENGTH;
        b += (int) hashes[j][i + 3];
        Set<Integer> set;
        if (buckets.containsKey(b)) {
          set = buckets.get(b);
          Set<Integer> cset;
          if (cands.containsKey(j)) {
            cset = cands.get(j);
          } else {
            cset = new HashSet<Integer>();
          }
          for (int s : set) {
            if (!cset.contains(s)) {
              cset.add(s);
            }
            Set<Integer> sset;
            if (cands.containsKey(s)) {
              sset = cands.get(s);
            } else {
              sset = new HashSet<Integer>();
            }
            sset.add(j);
            cands.put(s, sset);
          }
          cands.put(j, cset);
        } else {
          set = new HashSet<Integer>();
        }
        set.add(j);
        buckets.put(b, set);
      }
    }
    return cands;
  }
  private static String processQueries(Map<Integer, Set<Integer>> cands,
                                       Query[] qs, 
                                       byte[][] hashes, 
                                       String[] names, 
                                       Map<String, Integer> map) {
    int Q = qs.length;
    StringBuilder sb = new StringBuilder();
    double pr = 0.0;
    double rc = 0.0;
    int cnt = 0;
    for (int i = 0; i < Q; i++) {
      Query q = qs[i];
      int tid = q.getTextId();
      int hamm = q.getHammingDistance();
      if (cands.containsKey(tid)) {
        Set<Integer> cset = cands.get(tid);
        byte[] thash = hashes[tid];
        List<Integer> tids = new ArrayList<Integer>();
        //System.out.println("\t" + q + " : " + cset.size());
        int truep = 0;
        for (int cid : cset) {
          byte[] hash = hashes[cid];
          if (areNearDuplicates(thash, hash, hamm)) {
            tids.add(cid);
            if (names[tid].equals(names[cid])) {
              truep++;
            }
          }
        }
        Collections.sort(tids);
        int T = tids.size();
        /* for (int j = 0; j < T; j++) {
          sb.append(tids.get(j));
          if (j != T - 1) {
            sb.append(SP);
          }
        } */
        //sb.append(T);
        if (T != 0) {
          pr += (double) truep / (double) T;
          int all = map.get(names[tid]);
          rc += (double) truep / (double) (all - 1);
          
        } else {
          pr += 1.0;
          rc += 0.0;
        }
        cnt++;
      }
      //sb.append(NL);
    }
    DecimalFormat df = new DecimalFormat();
    df.setMaximumFractionDigits(2);
    pr /= (double) cnt;
    rc /= (double) cnt;
    sb.append("\t" + df.format(pr));
    sb.append("\t" + df.format(rc));
    sb.append("\t" + df.format((2 * (pr * rc) / (pr + rc))));
    //sb.append(NL);
    return sb.toString();
  }
  
  private static boolean areNearDuplicates(byte[] hash1, byte[] hash2, int h) {
    int hammingDistance = hammingDistance(hash1, hash2);
    if (hammingDistance <= h) return true;
    return false;
  }
  
  private static int hammingDistance(byte[] hash1, byte[] hash2) {
    int cnt = 0;
    int mask = 1;
    for (int i = 0; i < BAND_CNT; i++) {
      int and = hash1[i] ^ hash2[i];
      for (int j = 0; j < BAND_LENGTH; j++) {
        if ((mask & and >> j) == 1) cnt++;
      }
    }
    return cnt;
  }
  
  private static byte[] computeHash(String text) {
    int[] simHash = new int[HASH_LENGTH];
    int len = text.length();
    for (int i = 0; i < len - 5; i++) {
      simHash = computeHashWord(text.substring(i, i + 5), simHash);
    }
    byte[] wordHash = new byte[BAND_CNT];
    for (int i = 0; i < BAND_CNT; i++) {
      Integer byteVal = 0;
      for (int j = 0; j < BAND_LENGTH; j++) {
        byteVal <<= 1;
        if (simHash[(BAND_CNT -(i+1)) * BAND_LENGTH + (BAND_LENGTH - (j+1))] >= 0) {
          byteVal++;
        }
      }
      wordHash[i] = byteVal.byteValue();
    }
    return wordHash;
  }
  
  private static int[] computeHashWord(String word, int[] simHash) {
    byte[] wordHash;
    wordHash = DigestUtils.md5(word);
    for (int i = 0; i < BAND_CNT; i++) {
      int byteInt = (int) wordHash[i];
      int mask = 1;
      int cnt = 0;
      while (cnt < BAND_LENGTH) {
        if ((mask & byteInt >> cnt) ==  1) {
          simHash[(BAND_CNT - i - 1) * BAND_LENGTH + cnt]++;
        } else {
          simHash[(BAND_CNT - i - 1) * BAND_LENGTH + cnt]--;
        }
        cnt++;
      }
    }
    return simHash;
  }
  
  private static class Query {
    
    private int textId;
    private int hammingDistance;
    
    public Query(int qid, int tid, int pid, int d) {
      textId = tid;
      hammingDistance = d;
    }
    
    public int getTextId() { return textId; }
    public int getHammingDistance() { return hammingDistance; }
    
    @Override
    public String toString() {
      return "[" + textId + ", " + hammingDistance + "]";
    }
  }
}