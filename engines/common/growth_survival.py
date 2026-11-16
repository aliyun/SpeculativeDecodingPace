from .growth import KS


class PrefixSurvival:
    def __init__(self, rows, minimum=2, depth=7, *, batched_rows=False):
        accepted=[];side_hits=side_offers=0;rejected=0
        for row in rows:
            if row.get('terminal_censored',False) or not row.get('learning',True):continue
            k,roots,d,side=row['k'],row['branches'],row['delivered'],row['side_accepts']
            q=row['query_nodes'];offered=q-(k+1)
            # Structural corruption stays loud: an AR query must not pose as a native
            # chain, and a delivery cannot exceed its own tree width.
            if (type(k)is not int or k not in KS
                    or type(roots)is not int or roots not in(1,2)
                    or type(q)is not int or offered not in(0,1) or roots==1 and offered
                    or type(d)is not int or not 1<=d<=k+1
                    or type(side)is not int or side not in(0,1)):
                raise ValueError('GROWTH_SURVIVAL_FACTUAL')
            # Attribution ambiguity only costs us this row. Raising here runs inside a
            # live request, so an adapter that reported an offered side count as an
            # accepted one turned a bookkeeping mismatch into a 500 for the client.
            if (row.get('actual_batch',1)!=1 and not batched_rows
                    or side and(d!=2 or not offered)):
                rejected+=1
                continue
            a=d-1-side  # A selected alternate leaf means primary accepted zero.
            accepted.append((k,a))
            if offered:side_offers+=1;side_hits+=side
        self.rejected=rejected
        self.n=len(accepted);self.minimum=minimum
        self.survival=[];self.support=[];self.offered=[];self.risk=[];self.hits=[]
        previous=1.;ready=True
        for j in range(1,depth+1):
            offered=sum(k>=j for k,a in accepted)
            risk=sum(k>=j and a>=j-1 for k,a in accepted)
            hit=sum(k>=j and a>=j for k,a in accepted)
            if previous==0.:
                value=0.  # No invented positive tail beyond an estimated zero.
            elif previous is None or not risk:
                value=None
            else:
                value=previous*hit/risk
            ready=ready and value is not None and(offered>=minimum or previous==0.)
            self.survival.append(value);self.support.append(ready)
            self.offered.append(offered);self.risk.append(risk);self.hits.append(hit)
            previous=value
        self.side_offers=side_offers
        # Alternate-root success and primary-root success are disjoint events.
        self.side_rate=(min(side_hits/side_offers,1.-self.survival[0])
                        if side_offers and self.survival[0] is not None else None)
        self.side_supported=side_offers>=minimum

    def mean(self,k,remaining):
        count=min(k,remaining-1)
        values=self.survival[:count]
        return None if any(v is None for v in values) else 1.+sum(values)

    def supported(self,k,remaining):
        count=min(k,remaining-1)
        return not count or self.support[count-1]

    def optimistic_depth(self,k,remaining):
        """Extra tails cannot exceed survival at the deepest supported prefix.

        This uses an empirical head probability, NOT a statistical upper bound.
        It is only used alongside real measured action cost, never a fabricated
        cost for an action that has not yet been tried.
        """
        count=min(k,remaining-1);head=0
        for j in range(1,count+1):
            if not self.support[j-1]:break
            head=j
        if count and not head:return None
        base=self.mean(head,remaining)
        if base is None:return None
        reach=self.survival[head-1] if head else 1.
        return base+(count-head)*reach
