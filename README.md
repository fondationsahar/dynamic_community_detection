# Dynamic Community Detection

This repository compiles methods and experiments related to community structures in temporal networks.

The current code corresponds to the pre-print [*Longitudinal Modularity, a Modularity for Link Streams*](https://arxiv.org/abs/2408.16877).

#### Example 


To compute the Longitudinal Modularity of a dynamic community structure on a temporal network: 

```python
import longitudinal_modularity_tools as lmt


longitudinal_modularity = get_longitudinal_modularity(
    time_links_df=time_links_df, 
    # (pd.DataFrame) Interactions table. Columns: source, target, time, source_commu, target_commu
    communities_df=communities_df, 
    # (pd.DataFrame): Nodes communities memberships. Columns: node, time, commu
    expectation=expectation, 
    # (str, optional): Logitudinal expectation type. Must be "cm" for co-membership, "jm" for joint membership, or "mm" for mean membership. Defaults to "mm".
    omega=omega,
    # (float, optional): Time penalty weigth. High values will favor continuous communities. Defaults to 1.
)
```
