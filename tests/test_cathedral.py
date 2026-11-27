"""Cathedral composition, open geometry, replay, and agent integration."""
from copy import deepcopy
import tempfile
import unittest
from pathlib import Path
import numpy as np
import trimesh
from core.architecture import BuildRequest, BuildDocument, build, rebuild, describe_product
from core.architecture.cathedral import VARIANTS, random_controls
from core.architecture.agent import series_recipe, export_recipe


class CathedralTests(unittest.TestCase):
    def test_every_style_and_detail_builds_watertight_and_connected(self):
        silhouettes=[]
        for variant in VARIANTS:
            for detail in ('concept','standard','presentation'):
                with self.subTest(variant=variant,detail=detail):
                    r=build(BuildRequest(product='gothic-cathedral',variant=variant,detail=detail,seed=73))
                    self.assertEqual(r.quality.status,'pass',[c.to_dict() for c in r.quality.checks if c.status=='fail'])
                    self.assertTrue(all(b.mesh.is_watertight and b.mesh.is_winding_consistent and b.mesh.volume>0 for b in r.scene.blocks))
                    self.assertEqual(BuildDocument.from_dict(r.document.to_dict()).fingerprint(),r.document.fingerprint())
                    self.assertEqual(r.document.request.style,'gothic')
                    if detail=='concept':
                        silhouettes.append([(p.position,p.dimensions) for p in r.document.placements if p.role=='spire'])
        self.assertNotEqual(silhouettes[0],silhouettes[1])

    def test_replay_and_aliases_preserve_geometry_without_mutating_input(self):
        request=BuildRequest(product='  GOTHIC_CATHEDRAL ',variant='random',detail='concept',seed=5)
        before=deepcopy(request.to_dict())
        first=build(request);second=build(request)
        self.assertEqual(request.to_dict(),before)
        self.assertEqual(first.document.fingerprint(),second.document.fingerprint())
        replay=rebuild(BuildDocument.from_dict(first.document.to_dict()))
        self.assertEqual(replay.quality.status,'pass')
        for left,right in zip(first.scene.blocks,replay.scene.blocks):
            np.testing.assert_allclose(left.mesh.vertices,right.mesh.vertices,atol=1e-9)

    def test_damage_fails_even_after_mutable_rules_are_removed(self):
        r=build(BuildRequest(product='gothic-cathedral',detail='concept'))
        for damage,code in (('remove','cathedral.required_roles'),('detach','cathedral.envelope_connectivity'),('unbalance','cathedral.bilateral_composition')):
            doc=deepcopy(r.document);doc.metadata.clear();doc.request.product='GOTHIC_CATHEDRAL'
            if damage=='remove': doc.placements=[p for p in doc.placements if p.role!='vault_rib']
            else:
                p=next(p for p in doc.placements if p.role=='flying_buttress')
                p.position=(p.position[0]+(.5 if damage=='unbalance' else 100),p.position[1],p.position[2])
            report=rebuild(doc).quality
            self.assertIn(code,{c.code for c in report.checks if c.status=='fail'})

    def test_doorways_are_pierced_and_glazing_is_a_separate_asset(self):
        r=build(BuildRequest(product='gothic-cathedral',detail='concept'))
        portals=[b for b in r.scene.blocks if b.parameters['component_role']=='west_portal']
        groups={}
        for block in portals:
            groups.setdefault(block.parameters.get('opening_id'), []).append(block)
        self.assertGreaterEqual(len(groups), 5)
        for blocks in groups.values():
            # At sill level the combined surround has jambs on both sides and
            # no material at the central doorway, even when the arch is ashlars.
            v=np.vstack([b.mesh.vertices for b in blocks])
            bottom=v[:,1].min();height=np.ptp(v[:,1])
            axis=0 if np.ptp(v[:,0])>np.ptp(v[:,2]) else 2
            center=(v[:,axis].min()+v[:,axis].max())/2
            lower=v[v[:,1]<bottom+height*.2]
            self.assertGreater(np.min(np.abs(lower[:,axis]-center)),np.ptp(v[:,axis])*.25)
        panes=[b for b in r.scene.blocks if b.parameters['component_role']=='glazing']
        self.assertGreater(len(panes),20)
        self.assertTrue(all(b.color[3]<255 for b in panes))

    def test_design_envelope_and_discrete_controls(self):
        requests=[{'width':32,'height':40,'depth':52,'bays':4,'parameters':{'apse_sides':5,'choir_bays':2}},
                  {'width':32,'height':60,'depth':83,'bays':10,'parameters':{'apse_sides':9,'choir_bays':4,'spire_ratio':.36}}]
        for kwargs in requests:
            r=build(BuildRequest(product='gothic-cathedral',detail='concept',**kwargs))
            self.assertEqual(r.quality.status,'pass',[c.to_dict() for c in r.quality.checks if c.status=='fail'])
        for kwargs in ({'bays':3},{'width':32,'height':32},{'levels':2},{'openings':'none'},
                       {'parameters':{'apse_sides':6}},{'parameters':{'spire_ratio':True}},
                       {'parameters':{'unknown':1}},{'variant':'doric'}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                build(BuildRequest(product='gothic-cathedral',**kwargs))
        import random
        for seed in range(4):
            controls=random_controls(random.Random(seed))
            controls['detail']='concept'
            self.assertEqual(build(BuildRequest(product='gothic-cathedral',**controls)).quality.status,'pass')

    def test_agent_series_exports_real_cathedrals_and_replayable_members(self):
        recipe=series_recipe(2,73,products=('gothic-cathedral',),kitbash=False)
        for entry in recipe['buildings']:
            entry['request']['detail']='concept'
            entry['request']['width']=24
            entry['request']['height']=32
            entry['request']['depth']=42
        with tempfile.TemporaryDirectory() as tmp:
            receipt=export_recipe(recipe,Path(tmp)/'cathedrals')
            self.assertEqual(receipt['status'],'pass')
            self.assertEqual(receipt['count'],2)
            for record in receipt['buildings']:
                path=Path(receipt['output'])/record['artifacts']['glb']
                loaded=trimesh.load(path,force='scene')
                self.assertGreater(len(loaded.geometry),500)

    def test_catalog_descriptors_do_not_alias_authoritative_rules(self):
        descriptor=describe_product('gothic-cathedral')
        descriptor['rules']['required_roles'].clear()
        self.assertIn('vault_rib',describe_product('gothic-cathedral')['rules']['required_roles'])

    def test_concept_envelope_rose_and_spire_ribs_are_cubit_scale(self):
        from core.architecture.masonry import MAX_COURSE, MAX_STRETCHER
        result = build(BuildRequest(
            product='gothic-cathedral', detail='concept',
            width=32, height=40, depth=52, seed=73))
        self.assertEqual(result.quality.status, 'pass',
                         [c.to_dict() for c in result.quality.checks if c.status == 'fail'])
        self.assertLess(len(result.document.placements), 20000)
        roses = [p for p in result.document.placements if p.role == 'rose_tracery']
        self.assertGreater(len(roses), 20)
        self.assertGreater(len({p.parameters.get('host_id') for p in roses}), 3)
        oversized = []
        for placement in result.document.placements:
            if placement.role == 'glazing':
                continue
            width, height, depth = placement.dimensions
            if height > MAX_COURSE + 1e-6 or max(width, depth) > MAX_STRETCHER + 1e-6:
                oversized.append((placement.id, tuple(round(float(v), 3) for v in placement.dimensions)))
        self.assertEqual(oversized, [])
        for block in result.scene.blocks:
            if block.parameters.get("component_role") not in {"rose_tracery", "spire_rib"}:
                continue
            extents = block.mesh.bounds[1] - block.mesh.bounds[0]
            self.assertLessEqual(float(extents[1]), MAX_COURSE + 1e-6)
            self.assertLessEqual(max(float(extents[0]), float(extents[2])), MAX_STRETCHER + 1e-6)

    def test_presentation_default_stays_under_placement_cap(self):
        from core.architecture.cathedral import build_document
        from core.architecture.builtin import MAX_PLACEMENTS
        document = build_document(BuildRequest(product='gothic-cathedral', detail='presentation', seed=73))
        self.assertLess(len(document.placements), MAX_PLACEMENTS)

    def test_standard_tympanum_and_shafts_are_cubit_scale(self):
        from core.architecture.cathedral import build_document
        from core.architecture.masonry import MAX_COURSE, MAX_STRETCHER
        document = build_document(BuildRequest(
            product="gothic-cathedral", detail="standard",
            width=32, height=40, depth=52, seed=73))
        gables = [p for p in document.placements if p.role == "portal_gable"]
        self.assertGreater(len(gables), 20)
        oversized = []
        for placement in document.placements:
            if placement.role == "glazing":
                continue
            if placement.role not in {"portal_gable", "rose_tracery", "spire_rib", "buttress_cap",
                                      "finial", "pier_shaft", "tower_shaft"}:
                continue
            width, height, depth = placement.dimensions
            if height > MAX_COURSE + 1e-6 or max(width, depth) > MAX_STRETCHER + 1e-6:
                oversized.append((placement.id, tuple(round(float(v), 3) for v in placement.dimensions)))
        self.assertEqual(oversized, [])
